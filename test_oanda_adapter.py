"""Unit tests for OANDA broker adapter — Phase 3.

Self-contained (no conftest). Tests cover:
  - ForexHours (24/5 session logic)
  - Symbol normalization helpers
  - FOREX_MAJORS_AND_CROSSES constant
  - OandaBrokerAdapter (all 14 methods, mocked SDK)
  - OandaTickStreamer (init / properties)
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# ForexHours tests
# ===========================================================================


class TestForexHours:
    def setup_method(self):
        from brokers.market_hours import ForexHours
        self.fh = ForexHours()

    # -- is_market_day / is_in_session (equivalent for forex) ----------------

    def test_weekday_is_market_day(self):
        # Wednesday 2026-03-04 10:00
        now = datetime(2026, 3, 4, 10, 0)
        is_day, _ = self.fh.is_market_day(now)
        assert is_day

    def test_weekday_is_in_session(self):
        now = datetime(2026, 3, 4, 3, 0)  # Wednesday 3 AM
        assert self.fh.is_in_session(now)

    def test_saturday_not_market_day(self):
        now = datetime(2026, 3, 7, 12, 0)  # Saturday
        is_day, next_open = self.fh.is_market_day(now)
        assert not is_day
        assert next_open.weekday() == 6  # Sunday
        assert next_open.hour == 17

    def test_saturday_not_in_session(self):
        now = datetime(2026, 3, 7, 12, 0)  # Saturday
        assert not self.fh.is_in_session(now)

    def test_sunday_before_5pm_not_market_day(self):
        now = datetime(2026, 3, 8, 10, 0)  # Sunday 10 AM
        is_day, next_open = self.fh.is_market_day(now)
        assert not is_day
        assert next_open.hour == 17
        assert next_open.weekday() == 6  # Same Sunday

    def test_sunday_at_5pm_is_market_day(self):
        now = datetime(2026, 3, 8, 17, 0)  # Sunday 5 PM
        is_day, _ = self.fh.is_market_day(now)
        assert is_day

    def test_sunday_after_5pm_is_in_session(self):
        now = datetime(2026, 3, 8, 18, 0)  # Sunday 6 PM
        assert self.fh.is_in_session(now)

    def test_friday_before_4pm_is_market_day(self):
        now = datetime(2026, 3, 6, 15, 59)  # Friday 3:59 PM
        is_day, _ = self.fh.is_market_day(now)
        assert is_day

    def test_friday_at_4pm_not_market_day(self):
        now = datetime(2026, 3, 6, 16, 0)  # Friday 4 PM
        is_day, next_open = self.fh.is_market_day(now)
        assert not is_day
        assert next_open.weekday() == 6  # Sunday
        assert next_open.hour == 17

    def test_friday_after_4pm_not_in_session(self):
        now = datetime(2026, 3, 6, 16, 30)  # Friday 4:30 PM
        assert not self.fh.is_in_session(now)

    # -- is_entry_allowed ----------------------------------------------------

    def test_entry_allowed_weekday(self):
        now = datetime(2026, 3, 4, 10, 0)  # Wednesday
        ok, reason = self.fh.is_entry_allowed(now, 14, 0)
        assert ok
        assert reason == ''

    def test_entry_blocked_weekend(self):
        now = datetime(2026, 3, 7, 10, 0)  # Saturday
        ok, reason = self.fh.is_entry_allowed(now, 14, 0)
        assert not ok
        assert 'weekend' in reason.lower()

    def test_entry_blocked_friday_past_cutoff(self):
        now = datetime(2026, 3, 6, 14, 0)  # Friday 2 PM
        ok, reason = self.fh.is_entry_allowed(now, 14, 0)
        assert not ok
        assert 'Friday' in reason

    def test_entry_allowed_friday_before_cutoff(self):
        now = datetime(2026, 3, 6, 13, 59)  # Friday 1:59 PM
        ok, reason = self.fh.is_entry_allowed(now, 14, 0)
        assert ok

    # -- is_eod_close --------------------------------------------------------

    def test_eod_only_on_friday(self):
        # Wednesday — never triggers EOD
        now = datetime(2026, 3, 4, 15, 50)
        assert not self.fh.is_eod_close(now, 15, 50)

    def test_eod_friday_before(self):
        now = datetime(2026, 3, 6, 15, 49)  # Friday 15:49
        assert not self.fh.is_eod_close(now, 15, 50)

    def test_eod_friday_at(self):
        now = datetime(2026, 3, 6, 15, 50)  # Friday 15:50
        assert self.fh.is_eod_close(now, 15, 50)

    def test_eod_friday_after(self):
        now = datetime(2026, 3, 6, 15, 55)  # Friday 15:55
        assert self.fh.is_eod_close(now, 15, 50)


# ===========================================================================
# Symbol normalization tests
# ===========================================================================


class TestSymbolNormalization:
    def test_to_oanda(self):
        from brokers.oanda_adapter import _to_oanda
        assert _to_oanda('EUR/USD') == 'EUR_USD'
        assert _to_oanda('GBP/JPY') == 'GBP_JPY'

    def test_from_oanda(self):
        from brokers.oanda_adapter import _from_oanda
        assert _from_oanda('EUR_USD') == 'EUR/USD'
        assert _from_oanda('GBP_JPY') == 'GBP/JPY'

    def test_roundtrip(self):
        from brokers.oanda_adapter import _to_oanda, _from_oanda
        for pair in ['EUR/USD', 'GBP/CHF', 'AUD/NZD', 'CAD/JPY']:
            assert _from_oanda(_to_oanda(pair)) == pair


# ===========================================================================
# Default pairs constant tests
# ===========================================================================


class TestDefaultPairs:
    def test_majors_present(self):
        from brokers.oanda_adapter import FOREX_MAJORS_AND_CROSSES
        for major in ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD', 'USD/CAD']:
            assert major in FOREX_MAJORS_AND_CROSSES

    def test_slash_format(self):
        from brokers.oanda_adapter import FOREX_MAJORS_AND_CROSSES
        for pair in FOREX_MAJORS_AND_CROSSES:
            assert '/' in pair, f"{pair} missing slash"
            assert '_' not in pair, f"{pair} has underscore"

    def test_count_at_least_20(self):
        from brokers.oanda_adapter import FOREX_MAJORS_AND_CROSSES
        assert len(FOREX_MAJORS_AND_CROSSES) >= 20


# ===========================================================================
# OANDA interval map tests
# ===========================================================================


class TestIntervalMap:
    def test_common_intervals(self):
        from brokers.oanda_adapter import OANDA_INTERVAL_MAP
        assert OANDA_INTERVAL_MAP['1m'] == 'M1'
        assert OANDA_INTERVAL_MAP['1h'] == 'H1'
        assert OANDA_INTERVAL_MAP['1Day'] == 'D'
        assert OANDA_INTERVAL_MAP['1d'] == 'D'


# ===========================================================================
# Mock OANDA SDK helpers
# ===========================================================================


def _mock_env(token='test-token', account_id='101-001-12345-001', env='practice'):
    """Patch environment variables for OANDA config."""
    return patch.dict('os.environ', {
        'OANDA_TOKEN': token,
        'OANDA_ACCOUNT_ID': account_id,
        'OANDA_ENVIRONMENT': env,
    })


def _make_mock_api():
    """Create a mock oandapyV20.API object."""
    api = MagicMock()
    return api


@pytest.fixture
def oanda_adapter():
    """Create an OandaBrokerAdapter with mocked SDK."""
    from brokers.oanda_adapter import OandaBrokerAdapter
    with _mock_env():
        adapter = OandaBrokerAdapter()
        mock_api = _make_mock_api()
        adapter._api = mock_api
        # Clear cached env config so it re-reads from patched env
        adapter._env_config = None
        yield adapter, mock_api


# ===========================================================================
# OandaBrokerAdapter tests
# ===========================================================================


class TestOandaBrokerAdapter:
    def test_broker_id(self, oanda_adapter):
        adapter, _ = oanda_adapter
        assert adapter.broker_id == 'oanda'

    def test_is_configured_true(self):
        from brokers.oanda_adapter import OandaBrokerAdapter
        with _mock_env():
            adapter = OandaBrokerAdapter()
            adapter._env_config = None  # force re-read
            assert adapter.is_configured()

    def test_is_not_configured(self):
        from brokers.oanda_adapter import OandaBrokerAdapter
        with patch.dict('os.environ', {}, clear=True):
            adapter = OandaBrokerAdapter()
            adapter._env_config = None
            assert not adapter.is_configured()

    def test_is_not_configured_missing_account(self):
        from brokers.oanda_adapter import OandaBrokerAdapter
        with patch.dict('os.environ', {'OANDA_TOKEN': 'tok'}, clear=True):
            adapter = OandaBrokerAdapter()
            adapter._env_config = None
            assert not adapter.is_configured()

    @pytest.mark.asyncio
    async def test_get_account(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'account': {
                'NAV': '50000.00',
                'balance': '48000.00',
                'marginAvailable': '45000.00',
                'unrealizedPL': '200.00',
            }
        }
        acct = await adapter.get_account()
        assert acct is not None
        assert acct['equity'] == '50000.00'
        assert acct['balance'] == '48000.00'
        assert acct['buying_power'] == '45000.00'
        assert acct['broker_id'] == 'oanda'

    @pytest.mark.asyncio
    async def test_get_account_error(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("connection timeout")
        acct = await adapter.get_account()
        assert acct is None

    @pytest.mark.asyncio
    async def test_get_positions_long_and_short(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'positions': [
                {
                    'instrument': 'EUR_USD',
                    'long': {'units': '10000', 'averagePrice': '1.08500'},
                    'short': {'units': '0'},
                },
                {
                    'instrument': 'GBP_USD',
                    'long': {'units': '0'},
                    'short': {'units': '-5000', 'averagePrice': '1.26000'},
                },
            ]
        }
        positions = await adapter.get_positions()
        assert len(positions) == 2

        eur = positions[0]
        assert eur['symbol'] == 'EUR/USD'
        assert eur['qty'] == 10000.0
        assert eur['side'] == 'long'
        assert eur['broker_id'] == 'oanda'

        gbp = positions[1]
        assert gbp['symbol'] == 'GBP/USD'
        assert gbp['qty'] == 5000.0
        assert gbp['side'] == 'short'

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {'positions': []}
        positions = await adapter.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_error(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("connection error")
        positions = await adapter.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_submit_market_order(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'orderFillTransaction': {
                'id': '12345',
                'units': '10000',
                'price': '1.08550',
            }
        }
        result = await adapter.submit_order('EUR/USD', 10000, 'buy')
        assert result.is_filled
        assert result.order_id == '12345'
        assert result.filled_qty == 10000.0
        assert result.filled_avg_price == 1.08550
        assert result.symbol == 'EUR/USD'
        assert result.side == 'buy'

    @pytest.mark.asyncio
    async def test_submit_sell_order_signed_units(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'orderFillTransaction': {
                'id': '12346',
                'units': '-5000',
                'price': '1.26000',
            }
        }
        result = await adapter.submit_order('GBP/USD', 5000, 'sell')
        assert result.is_filled
        assert result.filled_qty == 5000.0  # abs value

        # Verify the API was called with negative units
        call_args = mock_api.request.call_args
        ep = call_args[0][0]
        assert ep.data['order']['units'] == '-5000'

    @pytest.mark.asyncio
    async def test_submit_order_error(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("insufficient margin")
        result = await adapter.submit_order('EUR/USD', 10000, 'buy')
        assert result.is_error
        assert 'insufficient margin' in result.error

    @pytest.mark.asyncio
    async def test_submit_limit_order_creates_pending(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        # First call: OrderCreate returns pending
        # Second call: OrderDetails returns filled
        mock_api.request.side_effect = [
            {
                'orderCreateTransaction': {'id': '99999'},
            },
            {
                'order': {
                    'id': '99999',
                    'state': 'FILLED',
                    'filledUnits': '10000',
                    'price': '1.08400',
                },
            },
        ]
        result = await adapter.submit_order(
            'EUR/USD', 10000, 'buy',
            order_type='limit', limit_price=1.08400, timeout_sec=5.0,
        )
        assert result.is_filled
        assert result.order_id == '99999'

    @pytest.mark.asyncio
    async def test_place_stop_order_short(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'orderCreateTransaction': {'id': 'stop-100'},
        }
        result = await adapter.place_stop_order(
            'EUR/USD', 10000, 1.09000, 0.003, 'short',
        )
        assert result['id'] == 'stop-100'
        assert result['status'] == 'new'
        assert 'error' not in result

        # Verify positive units (buy to close short)
        call_args = mock_api.request.call_args
        ep = call_args[0][0]
        assert ep.data['order']['units'] == '10000'

    @pytest.mark.asyncio
    async def test_place_stop_order_long(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'orderCreateTransaction': {'id': 'stop-101'},
        }
        result = await adapter.place_stop_order(
            'EUR/USD', 10000, 1.07000, 0.003, 'long',
        )
        assert result['id'] == 'stop-101'

        # Verify negative units (sell to close long)
        call_args = mock_api.request.call_args
        ep = call_args[0][0]
        assert ep.data['order']['units'] == '-10000'

    @pytest.mark.asyncio
    async def test_place_stop_order_error(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("bad request")
        result = await adapter.place_stop_order('EUR/USD', 10000, 1.09000)
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_cancel_order_success(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {'orderCancelTransaction': {'id': '123'}}
        ok = await adapter.cancel_order('123')
        assert ok is True

    @pytest.mark.asyncio
    async def test_cancel_order_failure(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("order not found")
        ok = await adapter.cancel_order('999')
        assert ok is False

    @pytest.mark.asyncio
    async def test_get_order(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'order': {
                'id': '456',
                'state': 'PENDING',
                'instrument': 'EUR_USD',
                'units': '10000',
                'price': '1.08500',
            }
        }
        result = await adapter.get_order('456')
        assert result is not None
        assert result['id'] == '456'
        assert result['status'] == 'new'  # PENDING → new
        assert result['symbol'] == 'EUR/USD'

    @pytest.mark.asyncio
    async def test_get_order_filled(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'order': {
                'id': '789',
                'state': 'FILLED',
                'instrument': 'GBP_USD',
                'units': '-5000',
                'price': '1.26000',
            }
        }
        result = await adapter.get_order('789')
        assert result['status'] == 'filled'

    @pytest.mark.asyncio
    async def test_get_order_not_found(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("order not found")
        result = await adapter.get_order('nonexistent')
        assert result is None

    @pytest.mark.asyncio
    async def test_replace_stop_order(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        # First call: cancel succeeds
        # Second call: new stop order created
        mock_api.request.side_effect = [
            {'orderCancelTransaction': {'id': 'old-stop'}},
            {'orderCreateTransaction': {'id': 'new-stop'}},
        ]
        result = await adapter.replace_stop_order(
            'old-stop', 'EUR/USD', 10000, 1.09500,
        )
        assert result['id'] == 'new-stop'
        assert mock_api.request.call_count == 2

    @pytest.mark.asyncio
    async def test_replace_stop_order_cancel_fails(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("order not found")
        result = await adapter.replace_stop_order(
            'bad-id', 'EUR/USD', 10000, 1.09500,
        )
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_check_tradeable(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'instruments': [
                {'name': 'EUR_USD', 'type': 'CURRENCY', 'displayName': 'EUR/USD'},
            ]
        }
        shortable, etb = await adapter.check_tradeable('EUR/USD')
        assert shortable is True
        assert etb is True

    @pytest.mark.asyncio
    async def test_check_tradeable_not_found(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {'instruments': []}
        shortable, etb = await adapter.check_tradeable('FAKE/PAIR')
        assert shortable is False
        assert etb is False

    @pytest.mark.asyncio
    async def test_get_tradeable_symbols(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'instruments': [
                {'name': 'EUR_USD', 'type': 'CURRENCY'},
                {'name': 'GBP_USD', 'type': 'CURRENCY'},
                {'name': 'SPX500_USD', 'type': 'CFD'},  # should be filtered
            ]
        }
        symbols = await adapter.get_tradeable_symbols()
        assert 'EUR/USD' in symbols
        assert 'GBP/USD' in symbols
        assert 'SPX500/USD' not in symbols  # CFD filtered out
        assert len(symbols) == 2

    @pytest.mark.asyncio
    async def test_get_tradeable_symbols_fallback(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.side_effect = Exception("network error")
        symbols = await adapter.get_tradeable_symbols()
        # Should fall back to FOREX_MAJORS_AND_CROSSES
        assert len(symbols) >= 20
        assert 'EUR/USD' in symbols

    @pytest.mark.asyncio
    async def test_fetch_bars(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'candles': [
                {
                    'time': '2026-02-02T00:00:00.000000000Z',
                    'complete': True,
                    'volume': 50000,
                    'mid': {'o': '1.08100', 'h': '1.08500', 'l': '1.07900', 'c': '1.08300'},
                },
                {
                    'time': '2026-02-03T00:00:00.000000000Z',
                    'complete': True,
                    'volume': 45000,
                    'mid': {'o': '1.08300', 'h': '1.08700', 'l': '1.08100', 'c': '1.08600'},
                },
                {
                    'time': '2026-02-04T00:00:00.000000000Z',
                    'complete': False,  # incomplete — should be skipped
                    'volume': 10000,
                    'mid': {'o': '1.08600', 'h': '1.08800', 'l': '1.08500', 'c': '1.08700'},
                },
            ]
        }
        df = await adapter.fetch_bars('EUR/USD', '2026-02-01', '2026-02-28')
        assert df is not None
        assert len(df) == 2  # incomplete candle skipped
        assert df.index.name == 'EUR/USD'
        assert 'close' in df.columns

    @pytest.mark.asyncio
    async def test_fetch_bars_empty(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {'candles': []}
        df = await adapter.fetch_bars('EUR/USD', '2026-02-01', '2026-02-28')
        assert df is None

    @pytest.mark.asyncio
    async def test_fetch_bars_unsupported_interval(self, oanda_adapter):
        adapter, _ = oanda_adapter
        df = await adapter.fetch_bars('EUR/USD', '2026-02-01', '2026-02-28', interval='2h')
        assert df is None

    @pytest.mark.asyncio
    async def test_fetch_bars_multi(self, oanda_adapter):
        adapter, mock_api = oanda_adapter
        mock_api.request.return_value = {
            'candles': [
                {
                    'time': '2026-02-02T00:00:00.000000000Z',
                    'complete': True,
                    'volume': 50000,
                    'mid': {'o': '1.08100', 'h': '1.08500', 'l': '1.07900', 'c': '1.08300'},
                },
            ]
        }
        result = await adapter.fetch_bars_multi(
            ['EUR/USD', 'GBP/USD'], '2026-02-01', '2026-02-28',
        )
        assert 'EUR/USD' in result
        assert 'GBP/USD' in result
        assert mock_api.request.call_count == 2

    def test_market_hours_returns_forex(self, oanda_adapter):
        from brokers.market_hours import ForexHours
        adapter, _ = oanda_adapter
        mh = adapter.get_market_hours()
        assert isinstance(mh, ForexHours)

    def test_create_tick_streamer(self, oanda_adapter):
        from brokers.oanda_adapter import OandaTickStreamer
        adapter, _ = oanda_adapter
        streamer = adapter.create_tick_streamer(['EUR/USD', 'GBP/USD'])
        assert isinstance(streamer, OandaTickStreamer)

    def test_create_tick_streamer_not_configured(self):
        from brokers.oanda_adapter import OandaBrokerAdapter
        with patch.dict('os.environ', {}, clear=True):
            adapter = OandaBrokerAdapter()
            adapter._env_config = None
            with pytest.raises(RuntimeError, match="not configured"):
                adapter.create_tick_streamer(['EUR/USD'])


# ===========================================================================
# OandaTickStreamer tests
# ===========================================================================


class TestOandaTickStreamer:
    def test_init(self):
        from brokers.oanda_adapter import OandaTickStreamer
        streamer = OandaTickStreamer(
            symbols=['EUR/USD', 'GBP/USD'],
            api_token='test-token',
            account_id='101-001-12345-001',
        )
        assert streamer.symbols == ['EUR/USD', 'GBP/USD']

    def test_symbols_property(self):
        from brokers.oanda_adapter import OandaTickStreamer
        streamer = OandaTickStreamer(
            symbols=['EUR/USD'],
            api_token='tok',
            account_id='acct',
        )
        assert 'EUR/USD' in streamer.symbols

    def test_latest_prices_initially_empty(self):
        from brokers.oanda_adapter import OandaTickStreamer
        streamer = OandaTickStreamer(
            symbols=['EUR/USD'],
            api_token='tok',
            account_id='acct',
        )
        assert streamer.latest_prices == {}

    def test_connected_default_false(self):
        from brokers.oanda_adapter import OandaTickStreamer
        streamer = OandaTickStreamer(
            symbols=['EUR/USD'],
            api_token='tok',
            account_id='acct',
        )
        assert streamer.connected is False


# ===========================================================================
# Import / export tests
# ===========================================================================


class TestBrokerPackageExports:
    def test_import_oanda_from_brokers(self):
        from brokers import OandaBrokerAdapter, OandaTickStreamer, ForexHours
        assert OandaBrokerAdapter is not None
        assert OandaTickStreamer is not None
        assert ForexHours is not None

    def test_all_exports(self):
        import brokers
        assert 'OandaBrokerAdapter' in brokers.__all__
        assert 'OandaTickStreamer' in brokers.__all__
        assert 'ForexHours' in brokers.__all__
