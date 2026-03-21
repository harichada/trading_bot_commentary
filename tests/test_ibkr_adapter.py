"""Unit tests for the IBKR broker adapter.

Tests IBKRConnectionManager, IBKRBrokerAdapter, and IBKRTickStreamer.
All ib_insync.IB interactions are mocked — no real gateway connection is made.

Coverage targets:
  - IBKRConnectionManager: connect, disconnect, qualify_contract, batch qualify, reconnect
  - IBKRBrokerAdapter: is_configured, get_account, get_positions, submit_order,
    place_stop_order, cancel_order, get_order, get_snapshot_prices, check_tradeable
  - IBKRTickStreamer: start, stop, add_symbols, remove_symbols, price update callback
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import time
from datetime import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Stub ib_insync BEFORE importing the broker modules.
# We import the real ib_insync data classes (Contract, Order, etc.) so that
# production code that instantiates them works correctly, but we mock IB()
# globally so no network calls are ever made.
# ---------------------------------------------------------------------------
from ib_insync import (
    Contract,
    LimitOrder,
    MarketOrder,
    Order,
    OrderStatus,
    Position,
    Stock,
    StopLimitOrder,
    Ticker,
    Trade,
)
from ib_insync.objects import AccountValue, TradeLogEntry

# Ensure brokers package is importable
_repo = os.path.dirname(os.path.dirname(__file__))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from brokers.base import BrokerConfig, OrderResult
from brokers.ibkr_connection import IBKRConnectionManager
from brokers.ibkr_streamer import IBKRTickStreamer
from brokers.ibkr_adapter import (
    IBKRBrokerAdapter,
    _interval_to_ibkr,
    _date_range_to_duration,
    _map_status,
)


# ---------------------------------------------------------------------------
# Helpers: build real ib_insync data objects for use in mocks
# ---------------------------------------------------------------------------

def _make_contract(symbol: str = "AAPL", con_id: int = 12345) -> Contract:
    c = Contract(symbol=symbol, conId=con_id, exchange="SMART", currency="USD")
    return c


def _make_order(order_id: int = 101, action: str = "BUY", qty: int = 10) -> Order:
    return Order(orderId=order_id, action=action, totalQuantity=qty)


def _make_order_status(
    order_id: int = 101,
    status: str = "Filled",
    filled: float = 10,
    avg_fill: float = 150.0,
    remaining: float = 0,
) -> OrderStatus:
    return OrderStatus(
        orderId=order_id,
        status=status,
        filled=filled,
        avgFillPrice=avg_fill,
        remaining=remaining,
    )


def _make_trade(
    symbol: str = "AAPL",
    order_id: int = 101,
    status: str = "Filled",
    filled: float = 10,
    avg_fill: float = 150.0,
    remaining: float = 0,
    log_messages: List[str] | None = None,
) -> Trade:
    contract = _make_contract(symbol)
    order = _make_order(order_id)
    os_ = _make_order_status(order_id, status, filled, avg_fill, remaining)
    log = []
    if log_messages:
        for msg in log_messages:
            log.append(TradeLogEntry(time=datetime.now(), status=status, message=msg))
    return Trade(contract=contract, order=order, orderStatus=os_, log=log)


def _make_ticker(symbol: str = "AAPL", price: float = 150.0) -> Ticker:
    """Return a Ticker with last price set so marketPrice() returns it."""
    contract = _make_contract(symbol)
    ticker = Ticker(contract=contract)
    ticker.last = price
    return ticker


def _make_account_value(tag: str, value: str) -> AccountValue:
    return AccountValue(account="DU123", tag=tag, value=value, currency="USD", modelCode="")


def _make_position(symbol: str, qty: float, avg_cost: float, con_id: int = 999) -> Position:
    contract = _make_contract(symbol, con_id)
    return Position(account="DU123", contract=contract, position=qty, avgCost=avg_cost)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ib() -> MagicMock:
    """Return a fully-configured MagicMock for ib_insync.IB()."""
    ib = MagicMock()
    # Default: not connected
    ib.isConnected.return_value = False
    # Make connectAsync awaitable
    ib.connectAsync = AsyncMock(return_value=None)
    # qualifyContractsAsync awaitable — returns empty by default
    ib.qualifyContractsAsync = AsyncMock(return_value=[])
    # accountSummaryAsync awaitable
    ib.accountSummaryAsync = AsyncMock(return_value=[])
    # reqContractDetailsAsync awaitable — returns a non-empty list by default
    # (non-empty means the contract is tradeable)
    ib.reqContractDetailsAsync = AsyncMock(return_value=[MagicMock()])
    # positions() sync
    ib.positions.return_value = []
    # openTrades() sync
    ib.openTrades.return_value = []
    # trades() sync
    ib.trades.return_value = []
    # placeOrder sync → returns a Trade
    ib.placeOrder.return_value = _make_trade()
    # cancelOrder sync
    ib.cancelOrder.return_value = None
    # reqMktData sync → returns a Ticker
    ib.reqMktData.return_value = _make_ticker()
    # cancelMktData sync
    ib.cancelMktData.return_value = None
    # reqMarketDataType sync
    ib.reqMarketDataType.return_value = None
    # Event-style attributes (+=/-= used in streamer)
    ib.disconnectedEvent = MagicMock()
    ib.pendingTickersEvent = MagicMock()
    # reqHistoricalDataAsync awaitable
    ib.reqHistoricalDataAsync = AsyncMock(return_value=[])
    return ib


@pytest.fixture
def conn(mock_ib: MagicMock) -> IBKRConnectionManager:
    """IBKRConnectionManager with its internal IB instance replaced by mock_ib."""
    mgr = IBKRConnectionManager(host="127.0.0.1", port=4002, client_id=1)
    mgr._ib = mock_ib
    return mgr


@pytest.fixture
def adapter(conn: IBKRConnectionManager) -> IBKRBrokerAdapter:
    """IBKRBrokerAdapter wired to the mocked connection manager."""
    cfg = BrokerConfig(broker_id="ibkr", environment="paper")
    return IBKRBrokerAdapter(config=cfg, conn=conn)


# ---------------------------------------------------------------------------
# IBKRConnectionManager Tests
# ---------------------------------------------------------------------------

class TestIBKRConnectionManager:

    # -- connect ------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_connect_success(self, conn: IBKRConnectionManager, mock_ib: MagicMock):
        """connect() returns True and sets _connected when gateway accepts."""
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(return_value=None)

        result = await conn.connect()

        assert result is True
        assert conn._connected is True
        mock_ib.connectAsync.assert_awaited_once_with(
            "127.0.0.1", 4002, 1, timeout=15.0
        )

    @pytest.mark.asyncio
    async def test_connect_already_connected_is_noop(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """connect() is a no-op and returns True if already connected."""
        mock_ib.isConnected.return_value = True

        result = await conn.connect()

        assert result is True
        mock_ib.connectAsync.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_connect_failure_returns_false(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """connect() returns False and clears _connected when gateway rejects."""
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("port closed"))

        result = await conn.connect()

        assert result is False
        assert conn._connected is False

    @pytest.mark.asyncio
    async def test_connect_timeout_returns_false(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """connect() returns False when connectAsync raises TimeoutError."""
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=asyncio.TimeoutError())

        result = await conn.connect()

        assert result is False

    # -- disconnect ---------------------------------------------------------

    @pytest.mark.asyncio
    async def test_disconnect_clears_cache(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """disconnect() clears contract cache and calls ib.disconnect()."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["SPY"] = _make_contract("SPY")
        conn._connected = True

        await conn.disconnect()

        mock_ib.disconnect.assert_called_once()
        assert conn._contract_cache == {}
        assert conn._connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected_does_not_call_ib(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """disconnect() skips ib.disconnect() when already disconnected."""
        mock_ib.isConnected.return_value = False

        await conn.disconnect()

        mock_ib.disconnect.assert_not_called()
        assert conn._connected is False

    # -- is_connected -------------------------------------------------------

    def test_is_connected_delegates_to_ib(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        assert conn.is_connected is True

        mock_ib.isConnected.return_value = False
        assert conn.is_connected is False

    # -- ensure_connected ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_ensure_connected_returns_true_if_already_connected(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        result = await conn.ensure_connected()
        assert result is True
        mock_ib.connectAsync.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ensure_connected_attempts_connect_if_disconnected(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(return_value=None)

        result = await conn.ensure_connected()

        assert result is True
        mock_ib.connectAsync.assert_awaited_once()

    # -- qualify_contract ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_qualify_contract_returns_qualified(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        qualified = _make_contract("AAPL", 12345)
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qualified])

        result = await conn.qualify_contract("AAPL")

        assert result is qualified
        assert conn._contract_cache["AAPL"] is qualified

    @pytest.mark.asyncio
    async def test_qualify_contract_uses_cache_on_second_call(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """qualify_contract() must not call the API again for cached symbols."""
        cached = _make_contract("SPY", 756733)
        conn._contract_cache["SPY"] = cached

        result = await conn.qualify_contract("SPY")

        assert result is cached
        mock_ib.qualifyContractsAsync.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_qualify_contract_returns_none_when_not_found(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])

        result = await conn.qualify_contract("NOTREAL")

        assert result is None

    @pytest.mark.asyncio
    async def test_qualify_contract_returns_none_on_exception(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.qualifyContractsAsync = AsyncMock(side_effect=RuntimeError("API error"))

        result = await conn.qualify_contract("ERR")

        assert result is None

    # -- qualify_contracts_batch --------------------------------------------

    @pytest.mark.asyncio
    async def test_qualify_contracts_batch_returns_multiple(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        spy = _make_contract("SPY", 756733)
        spy.symbol = "SPY"
        qqq = _make_contract("QQQ", 320227)
        qqq.symbol = "QQQ"
        spy.conId = 756733  # non-zero → qualified
        qqq.conId = 320227

        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[spy, qqq])

        result = await conn.qualify_contracts_batch(["SPY", "QQQ"])

        assert "SPY" in result
        assert "QQQ" in result

    @pytest.mark.asyncio
    async def test_qualify_contracts_batch_uses_cached_symbols(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """Symbols already cached skip the API call entirely."""
        cached_spy = _make_contract("SPY", 756733)
        conn._contract_cache["SPY"] = cached_spy

        qqq = _make_contract("QQQ", 320227)
        qqq.symbol = "QQQ"
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[qqq])

        result = await conn.qualify_contracts_batch(["SPY", "QQQ"])

        assert result["SPY"] is cached_spy
        assert "QQQ" in result
        # Only QQQ was sent to the API
        call_args = mock_ib.qualifyContractsAsync.await_args[0]
        assert len(call_args) == 1
        assert call_args[0].symbol == "QQQ"

    @pytest.mark.asyncio
    async def test_qualify_contracts_batch_empty_list(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        result = await conn.qualify_contracts_batch([])
        assert result == {}
        mock_ib.qualifyContractsAsync.assert_not_awaited()

    # -- _on_disconnect (reconnect scheduling) ------------------------------

    def test_on_disconnect_sets_flag(self, conn: IBKRConnectionManager):
        conn._connected = True
        conn._last_connect_attempt = time.monotonic() - 60  # long ago

        conn._on_disconnect()

        assert conn._connected is False

    def test_on_disconnect_skips_reconnect_if_too_recent(
        self, conn: IBKRConnectionManager
    ):
        """If we connected less than 5 s ago, suppress auto-reconnect."""
        conn._connected = True
        conn._last_connect_attempt = time.monotonic()  # just now

        # Should not raise and should not create a task
        conn._on_disconnect()

        assert conn._reconnect_task is None

    @pytest.mark.asyncio
    async def test_reconnect_loop_succeeds_on_first_retry(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """_reconnect_loop should stop once connect() returns True."""
        attempt_count = 0

        async def fake_connect_async(*args, **kwargs):
            nonlocal attempt_count
            attempt_count += 1
            mock_ib.isConnected.return_value = True

        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=fake_connect_async)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await conn._reconnect_loop()

        assert attempt_count == 1

    @pytest.mark.asyncio
    async def test_reconnect_loop_exhausts_all_attempts_on_failure(
        self, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """_reconnect_loop tries 5 times when every attempt fails."""
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError("down"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await conn._reconnect_loop()

        # 5 delay steps → 5 connect attempts
        assert mock_ib.connectAsync.await_count == 5


# ---------------------------------------------------------------------------
# IBKRBrokerAdapter Tests
# ---------------------------------------------------------------------------

class TestIBKRBrokerAdapter:

    # -- is_configured ------------------------------------------------------

    def test_is_configured_true_when_env_vars_set(self, adapter: IBKRBrokerAdapter):
        with patch.dict(os.environ, {"IBKR_HOST": "127.0.0.1", "IBKR_PORT": "4002"}):
            assert adapter.is_configured() is True

    def test_is_configured_false_when_env_vars_missing(self, adapter: IBKRBrokerAdapter):
        env = {"IBKR_HOST": "", "IBKR_PORT": ""}
        with patch.dict(os.environ, env, clear=False):
            # Temporarily remove both keys
            env_backup = {}
            for key in ("IBKR_HOST", "IBKR_PORT"):
                env_backup[key] = os.environ.pop(key, None)
            try:
                assert adapter.is_configured() is False
            finally:
                for key, val in env_backup.items():
                    if val is not None:
                        os.environ[key] = val

    def test_is_configured_false_when_port_missing(self, adapter: IBKRBrokerAdapter):
        with patch.dict(os.environ, {"IBKR_HOST": "127.0.0.1", "IBKR_PORT": ""}):
            # PORT is empty string → bool("") == False
            assert adapter.is_configured() is False

    # -- get_account --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_account_returns_parsed_values(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.accountSummaryAsync = AsyncMock(
            return_value=[
                _make_account_value("NetLiquidation", "100000"),
                _make_account_value("BuyingPower", "200000"),
                _make_account_value("TotalCashValue", "50000"),
                _make_account_value("GrossPositionValue", "80000"),
                _make_account_value("UnrealizedPnL", "1500"),
            ]
        )

        account = await adapter.get_account()

        assert account is not None
        assert account["equity"] == 100000.0
        assert account["buying_power"] == 200000.0
        assert account["cash"] == 50000.0
        assert account["portfolio_value"] == 80000.0
        assert account["unrealized_pnl"] == 1500.0
        assert account["broker_id"] == "ibkr"

    @pytest.mark.asyncio
    async def test_get_account_returns_none_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        result = await adapter.get_account()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_account_returns_none_on_api_error(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.accountSummaryAsync = AsyncMock(side_effect=RuntimeError("API error"))

        result = await adapter.get_account()

        assert result is None

    @pytest.mark.asyncio
    async def test_get_account_handles_missing_tags(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        """get_account() defaults to 0 when tags are absent."""
        mock_ib.isConnected.return_value = True
        mock_ib.accountSummaryAsync = AsyncMock(return_value=[])

        account = await adapter.get_account()

        assert account is not None
        assert account["equity"] == 0.0
        assert account["buying_power"] == 0.0

    # -- get_positions -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_positions_long(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [
            _make_position("AAPL", 100.0, 150.0, 12345),
        ]

        positions = await adapter.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos["symbol"] == "AAPL"
        assert pos["qty"] == 100.0
        assert pos["avg_entry_price"] == 150.0
        assert pos["side"] == "long"
        assert pos["broker_id"] == "ibkr"

    @pytest.mark.asyncio
    async def test_get_positions_short(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [
            _make_position("TSLA", -50.0, 200.0, 76792991),
        ]

        positions = await adapter.get_positions()

        assert len(positions) == 1
        pos = positions[0]
        assert pos["qty"] == 50.0  # abs value
        assert pos["side"] == "short"

    @pytest.mark.asyncio
    async def test_get_positions_skips_zero_qty(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.positions.return_value = [
            _make_position("FLAT", 0.0, 100.0),
        ]

        positions = await adapter.get_positions()

        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_empty_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        positions = await adapter.get_positions()

        assert positions == []

    @pytest.mark.asyncio
    async def test_get_positions_empty_on_api_error(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.positions.side_effect = RuntimeError("positions error")

        result = await adapter.get_positions()

        assert result == []

    # -- submit_order --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_submit_order_market_fill(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """submit_order with market type returns filled OrderResult immediately."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["AAPL"] = _make_contract("AAPL", 12345)
        filled_trade = _make_trade("AAPL", 101, "Filled", 10, 150.0)
        mock_ib.placeOrder.return_value = filled_trade

        result = await adapter.submit_order("AAPL", 10, "buy", order_type="market")

        assert result.status == "filled"
        assert result.filled_qty == 10.0
        assert result.filled_avg_price == 150.0
        assert result.symbol == "AAPL"
        assert result.side == "buy"
        assert result.order_id == "101"

    @pytest.mark.asyncio
    async def test_submit_order_limit_fills(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """submit_order with limit type places LimitOrder and returns filled."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["SPY"] = _make_contract("SPY", 756733)
        filled_trade = _make_trade("SPY", 202, "Filled", 5, 499.50)
        mock_ib.placeOrder.return_value = filled_trade

        result = await adapter.submit_order(
            "SPY", 5, "buy", order_type="limit", limit_price=500.0
        )

        assert result.status == "filled"
        # Verify a LimitOrder was placed (not MarketOrder)
        placed_order = mock_ib.placeOrder.call_args[0][1]
        assert isinstance(placed_order, LimitOrder)
        assert placed_order.lmtPrice == 500.0

    @pytest.mark.asyncio
    async def test_submit_order_sell_side(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """submit_order maps side='sell' to IBKR action='SELL'."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["MSFT"] = _make_contract("MSFT", 272093)
        sell_trade = _make_trade("MSFT", 303, "Filled", 20, 400.0)
        mock_ib.placeOrder.return_value = sell_trade

        result = await adapter.submit_order("MSFT", 20, "sell", order_type="market")

        placed_order = mock_ib.placeOrder.call_args[0][1]
        assert placed_order.action == "SELL"
        assert result.status == "filled"

    @pytest.mark.asyncio
    async def test_submit_order_short_maps_to_sell(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """Side='short' should map to IBKR action='SELL'."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["NVDA"] = _make_contract("NVDA", 4815747)
        trade = _make_trade("NVDA", 404, "Filled", 5, 900.0)
        mock_ib.placeOrder.return_value = trade

        await adapter.submit_order("NVDA", 5, "short", order_type="market")

        placed_order = mock_ib.placeOrder.call_args[0][1]
        assert placed_order.action == "SELL"

    @pytest.mark.asyncio
    async def test_submit_order_cancelled_returns_cancelled(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """When IBKR cancels the order, status is 'cancelled'."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["XYZ"] = _make_contract("XYZ", 999)
        cancelled_trade = _make_trade(
            "XYZ", 505, "Cancelled", 0, 0, 10, log_messages=["Order cancelled by user"]
        )
        mock_ib.placeOrder.return_value = cancelled_trade

        result = await adapter.submit_order("XYZ", 10, "buy")

        assert result.status == "cancelled"
        assert result.error != ""

    @pytest.mark.asyncio
    async def test_submit_order_inactive_returns_rejected(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        conn._contract_cache["JUNK"] = _make_contract("JUNK", 888)
        rejected_trade = _make_trade("JUNK", 606, "Inactive", 0, 0, 5)
        mock_ib.placeOrder.return_value = rejected_trade

        result = await adapter.submit_order("JUNK", 5, "buy")

        assert result.status == "rejected"

    @pytest.mark.asyncio
    async def test_submit_order_timeout_returns_pending(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """A limit order that never fills returns status='pending' after timeout."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["SLOW"] = _make_contract("SLOW", 777)
        # Trade stays 'Submitted' (never fills)
        pending_trade = _make_trade("SLOW", 707, "Submitted", 0, 0, 10)
        mock_ib.placeOrder.return_value = pending_trade

        # Use tiny timeout to keep test fast
        result = await adapter.submit_order(
            "SLOW", 10, "buy", order_type="limit", limit_price=100.0, timeout_sec=0.01
        )

        assert result.status == "pending"
        assert "Submitted" in result.error

    @pytest.mark.asyncio
    async def test_submit_order_partial_fill_on_timeout(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """A partially filled order on timeout returns status='partially_filled'."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["PART"] = _make_contract("PART", 666)
        partial_trade = _make_trade("PART", 808, "Submitted", 3, 100.0, 7)
        mock_ib.placeOrder.return_value = partial_trade

        result = await adapter.submit_order(
            "PART", 10, "buy", order_type="limit", limit_price=100.0, timeout_sec=0.01
        )

        assert result.status == "partially_filled"
        assert result.filled_qty == 3.0

    @pytest.mark.asyncio
    async def test_submit_order_error_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        result = await adapter.submit_order("AAPL", 10, "buy")

        assert result.status == "error"
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_submit_order_error_when_contract_not_qualified(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])

        result = await adapter.submit_order("FAKE", 10, "buy")

        assert result.status == "error"
        assert "FAKE" in result.error

    @pytest.mark.asyncio
    async def test_submit_order_error_on_placeorder_exception(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        conn._contract_cache["ERR"] = _make_contract("ERR", 555)
        mock_ib.placeOrder.side_effect = RuntimeError("gateway error")

        result = await adapter.submit_order("ERR", 10, "buy")

        assert result.status == "error"
        assert "gateway error" in result.error

    # -- place_stop_order ---------------------------------------------------

    @pytest.mark.asyncio
    async def test_place_stop_order_long_direction(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """Long position stop → SELL StopLimitOrder below stop_price."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["AAPL"] = _make_contract("AAPL", 12345)
        stop_trade = _make_trade("AAPL", 901, "Submitted", 0, 0, 10)
        stop_trade.orderStatus.status = "Submitted"
        mock_ib.placeOrder.return_value = stop_trade

        result = await adapter.place_stop_order(
            "AAPL", 10, stop_price=140.0, direction="long"
        )

        assert "error" not in result
        assert result["symbol"] == "AAPL"
        assert result["stop_price"] == 140.0
        # Limit should be slightly below stop
        assert result["limit_price"] < 140.0

        placed_order = mock_ib.placeOrder.call_args[0][1]
        assert isinstance(placed_order, StopLimitOrder)
        assert placed_order.action == "SELL"

    @pytest.mark.asyncio
    async def test_place_stop_order_short_direction(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """Short position stop → BUY StopLimitOrder above stop_price."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["TSLA"] = _make_contract("TSLA", 76792991)
        stop_trade = _make_trade("TSLA", 902, "Submitted", 0, 0, 5)
        mock_ib.placeOrder.return_value = stop_trade

        result = await adapter.place_stop_order(
            "TSLA", 5, stop_price=250.0, direction="short"
        )

        placed_order = mock_ib.placeOrder.call_args[0][1]
        assert placed_order.action == "BUY"
        assert result["limit_price"] > 250.0

    @pytest.mark.asyncio
    async def test_place_stop_order_error_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        result = await adapter.place_stop_order("AAPL", 10, 140.0)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_place_stop_order_error_on_unqualified_contract(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])

        result = await adapter.place_stop_order("BOGUS", 5, 100.0)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_place_stop_order_error_on_exception(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        conn._contract_cache["XYZ"] = _make_contract("XYZ", 999)
        mock_ib.placeOrder.side_effect = RuntimeError("stop order rejected")

        result = await adapter.place_stop_order("XYZ", 5, 100.0)

        assert "error" in result
        assert "stop order rejected" in result["error"]

    # -- cancel_order -------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cancel_order_success(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        open_trade = _make_trade("AAPL", 123, "Submitted", 0, 0, 10)
        mock_ib.openTrades.return_value = [open_trade]

        result = await adapter.cancel_order("123")

        assert result is True
        mock_ib.cancelOrder.assert_called_once_with(open_trade.order)

    @pytest.mark.asyncio
    async def test_cancel_order_not_found(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.openTrades.return_value = []

        result = await adapter.cancel_order("999")

        assert result is False
        mock_ib.cancelOrder.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_order_false_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        result = await adapter.cancel_order("456")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_order_false_on_exception(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.openTrades.side_effect = RuntimeError("trades error")

        result = await adapter.cancel_order("789")

        assert result is False

    # -- get_order ----------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_order_found(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        trade = _make_trade("AAPL", 101, "Filled", 10, 150.0, 0)
        mock_ib.trades.return_value = [trade]

        result = await adapter.get_order("101")

        assert result is not None
        assert result["id"] == "101"
        assert result["status"] == "filled"
        assert result["ibkr_status"] == "Filled"
        assert result["filled_qty"] == 10.0
        assert result["avg_fill_price"] == 150.0
        assert result["symbol"] == "AAPL"

    @pytest.mark.asyncio
    async def test_get_order_pending_status(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        trade = _make_trade("SPY", 202, "Submitted", 0, 0, 5)
        mock_ib.trades.return_value = [trade]

        result = await adapter.get_order("202")

        assert result["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_order_not_found_returns_none(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.trades.return_value = []

        result = await adapter.get_order("404")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_order_none_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        result = await adapter.get_order("101")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_order_none_on_exception(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.trades.side_effect = RuntimeError("api error")

        result = await adapter.get_order("101")

        assert result is None

    # -- get_snapshot_prices ------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_snapshot_prices_returns_prices(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True

        spy_contract = _make_contract("SPY", 756733)
        spy_contract.symbol = "SPY"
        qqq_contract = _make_contract("QQQ", 320227)
        qqq_contract.symbol = "QQQ"

        conn._contract_cache["SPY"] = spy_contract
        conn._contract_cache["QQQ"] = qqq_contract

        spy_ticker = _make_ticker("SPY", 500.0)
        qqq_ticker = _make_ticker("QQQ", 430.0)

        mock_ib.reqMktData.side_effect = [spy_ticker, qqq_ticker]

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.get_snapshot_prices(["SPY", "QQQ"])

        assert result["SPY"] == 500.0
        assert result["QQQ"] == 430.0

    @pytest.mark.asyncio
    async def test_get_snapshot_prices_empty_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.get_snapshot_prices(["AAPL"])

        assert result == {}

    @pytest.mark.asyncio
    async def test_get_snapshot_prices_skips_nan_price(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """Symbols with NaN/zero price are excluded from results."""
        mock_ib.isConnected.return_value = True

        aapl_contract = _make_contract("AAPL", 12345)
        aapl_contract.symbol = "AAPL"
        conn._contract_cache["AAPL"] = aapl_contract

        # Ticker with no valid price (all NaN/zero)
        nan_ticker = Ticker(contract=aapl_contract)
        # last is NaN, close is NaN → no price
        mock_ib.reqMktData.return_value = nan_ticker

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.get_snapshot_prices(["AAPL"])

        assert "AAPL" not in result

    @pytest.mark.asyncio
    async def test_get_snapshot_prices_falls_back_to_last(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """When marketPrice() is NaN but last > 0, use last."""
        mock_ib.isConnected.return_value = True

        aapl_contract = _make_contract("AAPL", 12345)
        aapl_contract.symbol = "AAPL"
        conn._contract_cache["AAPL"] = aapl_contract

        ticker = Ticker(contract=aapl_contract)
        # marketPrice() returns NaN by default; set last to valid price
        ticker.last = 175.0

        mock_ib.reqMktData.return_value = ticker

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.get_snapshot_prices(["AAPL"])

        assert result.get("AAPL") == 175.0

    # -- check_tradeable ----------------------------------------------------

    @pytest.mark.asyncio
    async def test_check_tradeable_shortable_easy_to_borrow(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """shortableShares > 1M → (True, True)."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["SPY"] = _make_contract("SPY", 756733)
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[MagicMock()])

        ticker = _make_ticker("SPY", 500.0)
        ticker.shortableShares = 5_000_000.0
        mock_ib.reqMktData.return_value = ticker

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("SPY")

        assert shortable is True
        assert easy is True

    @pytest.mark.asyncio
    async def test_check_tradeable_shortable_hard_to_borrow(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """0 < shortableShares <= 1M → (True, False)."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["MEME"] = _make_contract("MEME", 11111)
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[MagicMock()])

        ticker = _make_ticker("MEME", 20.0)
        ticker.shortableShares = 500_000.0
        mock_ib.reqMktData.return_value = ticker

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("MEME")

        assert shortable is True
        assert easy is False

    @pytest.mark.asyncio
    async def test_check_tradeable_false_when_no_contract_details(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """When reqContractDetailsAsync returns empty, contract is not tradeable."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["DELISTED"] = _make_contract("DELISTED", 55555)
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("DELISTED")

        assert shortable is False
        assert easy is False

    @pytest.mark.asyncio
    async def test_check_tradeable_not_shortable(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        """shortableShares <= 0 → tradeable but not shortable: (True, False)."""
        mock_ib.isConnected.return_value = True
        conn._contract_cache["NOSHT"] = _make_contract("NOSHT", 22222)
        mock_ib.reqContractDetailsAsync = AsyncMock(return_value=[MagicMock()])

        ticker = _make_ticker("NOSHT", 50.0)
        ticker.shortableShares = 0.0
        mock_ib.reqMktData.return_value = ticker

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("NOSHT")

        assert shortable is True   # contract exists → tradeable
        assert easy is False

    @pytest.mark.asyncio
    async def test_check_tradeable_false_when_disconnected(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("AAPL")

        assert shortable is False
        assert easy is False

    @pytest.mark.asyncio
    async def test_check_tradeable_false_when_contract_unqualified(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("FAKE")

        assert shortable is False

    @pytest.mark.asyncio
    async def test_check_tradeable_false_on_exception(
        self, adapter: IBKRBrokerAdapter, conn: IBKRConnectionManager, mock_ib: MagicMock
    ):
        mock_ib.isConnected.return_value = True
        conn._contract_cache["ERR"] = _make_contract("ERR", 33333)
        # Trigger exception inside the try block
        mock_ib.reqContractDetailsAsync = AsyncMock(side_effect=RuntimeError("data error"))

        with patch("asyncio.sleep", new_callable=AsyncMock):
            shortable, easy = await adapter.check_tradeable("ERR")

        assert shortable is False

    # -- create_tick_streamer -----------------------------------------------

    def test_create_tick_streamer_returns_ibkr_streamer(
        self, adapter: IBKRBrokerAdapter
    ):
        streamer = adapter.create_tick_streamer(["AAPL", "MSFT"])
        assert isinstance(streamer, IBKRTickStreamer)
        assert "AAPL" in streamer.symbols
        assert "MSFT" in streamer.symbols

    def test_create_tick_streamer_passes_on_tick_callback(
        self, adapter: IBKRBrokerAdapter
    ):
        ticks = []
        on_tick = lambda s, p, t: ticks.append((s, p))
        streamer = adapter.create_tick_streamer(["SPY"], on_tick=on_tick)
        assert streamer._on_tick is on_tick

    # -- replace_stop_order --------------------------------------------------

    @pytest.mark.asyncio
    async def test_replace_stop_order_cancels_old_and_places_new(
        self, adapter: IBKRBrokerAdapter, mock_ib: MagicMock,
        conn: IBKRConnectionManager
    ):
        mock_ib.isConnected.return_value = True
        conn._contract_cache["AAPL"] = _make_contract("AAPL", 12345)

        old_trade = _make_trade("AAPL", 100, "Submitted", 0, 0, 10)
        new_stop_trade = _make_trade("AAPL", 101, "Submitted", 0, 0, 10)
        mock_ib.openTrades.return_value = [old_trade]
        mock_ib.placeOrder.return_value = new_stop_trade

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await adapter.replace_stop_order(
                "100", "AAPL", 10, 145.0
            )

        mock_ib.cancelOrder.assert_called_once()
        assert "error" not in result

    # -- get_market_hours ----------------------------------------------------

    def test_get_market_hours_returns_us_equity_hours(
        self, adapter: IBKRBrokerAdapter
    ):
        from brokers.market_hours import USEquityHours
        hours = adapter.get_market_hours()
        assert isinstance(hours, USEquityHours)

    # -- get_tradeable_symbols -----------------------------------------------

    @pytest.mark.asyncio
    async def test_get_tradeable_symbols_returns_empty(
        self, adapter: IBKRBrokerAdapter
    ):
        """IBKR has no listing API — always returns empty list."""
        result = await adapter.get_tradeable_symbols()
        assert result == []


# ---------------------------------------------------------------------------
# IBKRTickStreamer Tests
# ---------------------------------------------------------------------------

class TestIBKRTickStreamer:
    """Tests for the real-time tick streamer."""

    @pytest.fixture
    def mock_conn(self, mock_ib: MagicMock) -> IBKRConnectionManager:
        mgr = IBKRConnectionManager(host="127.0.0.1", port=4002, client_id=1)
        mgr._ib = mock_ib
        return mgr

    @pytest.fixture
    def streamer(self, mock_conn: IBKRConnectionManager) -> IBKRTickStreamer:
        return IBKRTickStreamer(["AAPL", "MSFT"], mock_conn)

    # -- start --------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_start_subscribes_to_qualified_symbols(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = True

        aapl = _make_contract("AAPL", 12345)
        aapl.symbol = "AAPL"
        msft = _make_contract("MSFT", 272093)
        msft.symbol = "MSFT"
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[aapl, msft])

        aapl_ticker = _make_ticker("AAPL", 150.0)
        msft_ticker = _make_ticker("MSFT", 400.0)
        mock_ib.reqMktData.side_effect = [aapl_ticker, msft_ticker]

        await streamer.start()

        assert streamer.connected is True
        assert mock_ib.reqMktData.call_count == 2
        assert "AAPL" in streamer._tickers
        assert "MSFT" in streamer._tickers

    @pytest.mark.asyncio
    async def test_start_registers_pending_tickers_event(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        """start() attaches _on_pending_tickers to pendingTickersEvent via +=.

        Python compiles ``event += handler`` as
        ``event = event.__iadd__(handler)``, so we capture the ORIGINAL
        event mock before start() runs, then assert __iadd__ was called on it.
        """
        mock_ib.isConnected.return_value = True
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[])

        # Capture the original mock BEFORE start() so we can check __iadd__.
        original_event = mock_ib.pendingTickersEvent

        await streamer.start()

        original_event.__iadd__.assert_called_once_with(streamer._on_pending_tickers)

    @pytest.mark.asyncio
    async def test_start_aborts_when_not_connected(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = False
        mock_ib.connectAsync = AsyncMock(side_effect=ConnectionRefusedError())

        await streamer.start()

        assert streamer.connected is False
        mock_ib.reqMktData.assert_not_called()

    # -- stop ---------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stop_cancels_all_subscriptions(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = True
        # Pre-populate tickers as if start() was called
        streamer._tickers["AAPL"] = _make_ticker("AAPL", 150.0)
        streamer._tickers["MSFT"] = _make_ticker("MSFT", 400.0)
        streamer._connected = True

        await streamer.stop()

        assert mock_ib.cancelMktData.call_count == 2
        assert streamer._tickers == {}
        assert streamer.connected is False

    @pytest.mark.asyncio
    async def test_stop_when_disconnected_clears_state(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = False
        streamer._tickers["AAPL"] = _make_ticker("AAPL", 150.0)
        streamer._connected = True

        await streamer.stop()

        assert streamer._tickers == {}
        assert streamer.connected is False
        mock_ib.cancelMktData.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_handles_cancel_exception_gracefully(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        """cancelMktData exception should not propagate."""
        mock_ib.isConnected.return_value = True
        streamer._tickers["AAPL"] = _make_ticker("AAPL", 150.0)
        streamer._connected = True
        mock_ib.cancelMktData.side_effect = RuntimeError("cancel failed")

        # Should not raise
        await streamer.stop()

        assert streamer.connected is False

    # -- add_symbols --------------------------------------------------------

    @pytest.mark.asyncio
    async def test_add_symbols_subscribes_to_new_symbol(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = True
        nvda = _make_contract("NVDA", 4815747)
        nvda.symbol = "NVDA"
        mock_ib.qualifyContractsAsync = AsyncMock(return_value=[nvda])
        nvda_ticker = _make_ticker("NVDA", 900.0)
        mock_ib.reqMktData.return_value = nvda_ticker

        await streamer.add_symbols(["NVDA"])

        assert "NVDA" in streamer._tickers
        assert "NVDA" in streamer._symbols

    @pytest.mark.asyncio
    async def test_add_symbols_skips_already_subscribed(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = True
        streamer._tickers["AAPL"] = _make_ticker("AAPL", 150.0)

        await streamer.add_symbols(["AAPL"])

        mock_ib.qualifyContractsAsync.assert_not_awaited()
        mock_ib.reqMktData.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_symbols_buffers_when_disconnected(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        """When not connected, new symbols are appended to _symbols list only."""
        mock_ib.isConnected.return_value = False
        initial_count = len(streamer._symbols)

        await streamer.add_symbols(["GOOGL"])

        assert "GOOGL" in streamer._symbols
        assert len(streamer._tickers) == 0

    @pytest.mark.asyncio
    async def test_add_symbols_respects_max_limit(
        self,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        """Adding symbols beyond MAX_SYMBOLS limit should be silently rejected."""
        # Start with MAX_SYMBOLS already subscribed
        many_symbols = [f"SYM{i:03d}" for i in range(IBKRTickStreamer.MAX_SYMBOLS)]
        streamer = IBKRTickStreamer([], mock_conn)
        for sym in many_symbols:
            streamer._tickers[sym] = _make_ticker(sym, 10.0)

        mock_ib.isConnected.return_value = True

        await streamer.add_symbols(["NEWONE"])

        # NEWONE should not be added
        assert "NEWONE" not in streamer._tickers
        mock_ib.qualifyContractsAsync.assert_not_awaited()

    # -- remove_symbols -----------------------------------------------------

    @pytest.mark.asyncio
    async def test_remove_symbols_unsubscribes(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = True
        aapl_ticker = _make_ticker("AAPL", 150.0)
        streamer._tickers["AAPL"] = aapl_ticker
        streamer._symbols = ["AAPL", "MSFT"]

        await streamer.remove_symbols(["AAPL"])

        mock_ib.cancelMktData.assert_called_once()
        assert "AAPL" not in streamer._tickers
        assert "AAPL" not in streamer._symbols

    @pytest.mark.asyncio
    async def test_remove_symbols_skips_unknown(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        """Removing a symbol that was never subscribed should not error."""
        mock_ib.isConnected.return_value = True

        await streamer.remove_symbols(["UNKNOWN"])

        mock_ib.cancelMktData.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_symbols_when_disconnected_updates_list_only(
        self,
        streamer: IBKRTickStreamer,
        mock_conn: IBKRConnectionManager,
        mock_ib: MagicMock,
    ):
        mock_ib.isConnected.return_value = False
        streamer._symbols = ["AAPL", "MSFT", "GOOGL"]

        await streamer.remove_symbols(["AAPL", "MSFT"])

        assert "AAPL" not in streamer._symbols
        assert "MSFT" not in streamer._symbols
        assert "GOOGL" in streamer._symbols
        mock_ib.cancelMktData.assert_not_called()

    # -- _on_pending_tickers ------------------------------------------------

    def test_on_pending_tickers_updates_latest_prices(self, streamer: IBKRTickStreamer):
        """_on_pending_tickers should update _latest_prices for valid prices."""
        ticker = _make_ticker("AAPL", 155.0)
        # marketPrice() returns last when last is set

        streamer._on_pending_tickers([ticker])

        assert streamer._latest_prices["AAPL"] == 155.0

    def test_on_pending_tickers_ignores_zero_price(self, streamer: IBKRTickStreamer):
        """Tickers with price=0 should be discarded."""
        ticker = Ticker(contract=_make_contract("ZERO"))
        ticker.last = 0.0

        streamer._on_pending_tickers([ticker])

        assert "ZERO" not in streamer._latest_prices

    def test_on_pending_tickers_ignores_nan_price(self, streamer: IBKRTickStreamer):
        """Tickers with NaN price and no fallback should be discarded."""
        ticker = Ticker(contract=_make_contract("NAN"))
        # last/close remain NaN

        streamer._on_pending_tickers([ticker])

        assert "NAN" not in streamer._latest_prices

    def test_on_pending_tickers_throttles_rapid_updates(
        self, streamer: IBKRTickStreamer
    ):
        """Updates within THROTTLE_SEC should be suppressed."""
        ticker = _make_ticker("AAPL", 100.0)
        # Simulate first update
        streamer._on_pending_tickers([ticker])
        assert streamer._latest_prices["AAPL"] == 100.0

        # Immediately update again with a different price
        ticker2 = _make_ticker("AAPL", 200.0)
        streamer._on_pending_tickers([ticker2])

        # Still 100.0 because throttle window hasn't expired
        assert streamer._latest_prices["AAPL"] == 100.0

    def test_on_pending_tickers_allows_update_after_throttle(
        self, streamer: IBKRTickStreamer
    ):
        """After THROTTLE_SEC has elapsed, updates should pass through."""
        ticker = _make_ticker("AAPL", 100.0)
        streamer._on_pending_tickers([ticker])

        # Force last-push time to be old enough
        streamer._last_push["AAPL"] = time.monotonic() - (IBKRTickStreamer.THROTTLE_SEC + 0.1)

        ticker2 = _make_ticker("AAPL", 200.0)
        streamer._on_pending_tickers([ticker2])

        assert streamer._latest_prices["AAPL"] == 200.0

    def test_on_pending_tickers_calls_on_tick_callback(
        self, streamer: IBKRTickStreamer
    ):
        """_on_tick callback should be invoked with (symbol, price, timestamp)."""
        ticks_received = []
        streamer._on_tick = lambda s, p, t: ticks_received.append((s, p))

        ticker = _make_ticker("AAPL", 160.0)
        streamer._on_pending_tickers([ticker])

        assert len(ticks_received) == 1
        assert ticks_received[0] == ("AAPL", 160.0)

    def test_on_pending_tickers_survives_callback_exception(
        self, streamer: IBKRTickStreamer
    ):
        """Exception inside on_tick must not propagate to ib_insync."""
        streamer._on_tick = lambda s, p, t: (_ for _ in ()).throw(RuntimeError("cb error"))

        ticker = _make_ticker("AAPL", 160.0)
        # Should not raise
        streamer._on_pending_tickers([ticker])

        # Price was still recorded
        assert streamer._latest_prices["AAPL"] == 160.0

    def test_on_pending_tickers_falls_back_to_close_price(
        self, streamer: IBKRTickStreamer
    ):
        """When marketPrice() is NaN but close > 0, close should be used."""
        contract = _make_contract("CLOSE", 44444)
        ticker = Ticker(contract=contract)
        # leave last as NaN, set close
        ticker.close = 88.0

        streamer._on_pending_tickers([ticker])

        assert streamer._latest_prices["CLOSE"] == 88.0

    # -- latest_prices / symbols / connected properties --------------------

    def test_latest_prices_returns_copy(self, streamer: IBKRTickStreamer):
        """latest_prices must return a copy, not the internal dict."""
        streamer._latest_prices["X"] = 1.0
        prices = streamer.latest_prices
        prices["X"] = 999.0
        assert streamer._latest_prices["X"] == 1.0

    def test_symbols_returns_copy(self, streamer: IBKRTickStreamer):
        syms = streamer.symbols
        syms.append("HACKED")
        assert "HACKED" not in streamer._symbols

    def test_connected_property(self, streamer: IBKRTickStreamer):
        streamer._connected = True
        assert streamer.connected is True
        streamer._connected = False
        assert streamer.connected is False


# ---------------------------------------------------------------------------
# Helper function tests (_interval_to_ibkr, _date_range_to_duration, _map_status)
# ---------------------------------------------------------------------------

class TestHelperFunctions:

    @pytest.mark.parametrize("interval, expected", [
        ("1min", "1 min"),
        ("1Min", "1 min"),
        ("5min", "5 mins"),
        ("5Min", "5 mins"),
        ("15min", "15 mins"),
        ("15Min", "15 mins"),
        ("1hour", "1 hour"),
        ("1Hour", "1 hour"),
        ("1day", "1 day"),
        ("1Day", "1 day"),
        ("unknown", "1 day"),    # unknown defaults to 1 day
        ("", "1 day"),
    ])
    def test_interval_to_ibkr(self, interval: str, expected: str):
        assert _interval_to_ibkr(interval) == expected

    @pytest.mark.parametrize("start, end, expected_suffix", [
        ("2024-01-01", "2024-01-03", "D"),   # 3 days  → "3 D"
        ("2024-01-01", "2024-01-30", "D"),   # 30 days → "30 D" (boundary: <= 30)
        ("2024-01-01", "2024-04-01", "M"),   # ~91 days → months
        ("2023-01-01", "2024-01-01", "Y"),   # ~365 days → "1 Y"
        ("2020-01-01", "2024-01-01", "Y"),   # ~4 years → "4 Y"
    ])
    def test_date_range_to_duration_suffix(self, start: str, end: str, expected_suffix: str):
        result = _date_range_to_duration(start, end)
        assert result.endswith(expected_suffix), f"Got {result!r}, expected suffix {expected_suffix!r}"

    def test_date_range_to_duration_invalid_dates(self):
        """Invalid dates fall back to '6 M'."""
        result = _date_range_to_duration("bad-date", "also-bad")
        assert result == "6 M"

    @pytest.mark.parametrize("ibkr_status, canonical", [
        ("Filled", "filled"),
        ("Cancelled", "cancelled"),
        ("ApiCancelled", "cancelled"),
        ("PendingCancel", "cancelled"),
        ("Inactive", "rejected"),
        ("Submitted", "pending"),
        ("PreSubmitted", "pending"),
        ("PendingSubmit", "pending"),
        ("SomethingWeird", "error"),
        ("", "error"),
    ])
    def test_map_status(self, ibkr_status: str, canonical: str):
        assert _map_status(ibkr_status) == canonical


# ---------------------------------------------------------------------------
# OrderResult property tests
# ---------------------------------------------------------------------------

class TestOrderResult:

    def test_is_filled(self):
        r = OrderResult(status="filled")
        assert r.is_filled is True

    def test_is_not_filled(self):
        for status in ("error", "pending", "cancelled", "rejected", "partially_filled"):
            r = OrderResult(status=status)
            assert r.is_filled is False, f"Expected is_filled=False for {status!r}"

    @pytest.mark.parametrize("status", ["error", "rejected", "cancelled", "timeout"])
    def test_is_error(self, status: str):
        r = OrderResult(status=status)
        assert r.is_error is True

    @pytest.mark.parametrize("status", ["filled", "pending", "partially_filled"])
    def test_is_not_error(self, status: str):
        r = OrderResult(status=status)
        assert r.is_error is False
