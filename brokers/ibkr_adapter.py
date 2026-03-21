"""IBKR broker adapter — implements AbstractBroker using ib_insync.

Connects to IB Gateway (TWS API) for order execution, account management,
and real-time market data. Designed for US equities via SMART routing.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from ib_insync import (
    LimitOrder,
    MarketOrder,
    Order,
    Stock,
    StopLimitOrder,
    Trade,
)

from brokers.base import AbstractBroker, BrokerConfig, OrderResult, TickStreamer
from brokers.ibkr_connection import IBKRConnectionManager
from brokers.ibkr_streamer import IBKRTickStreamer
from brokers.market_hours import MarketHoursPolicy, USEquityHours

logger = logging.getLogger(__name__)

# IBKR order status → canonical status mapping
_STATUS_MAP = {
    'PendingSubmit': 'pending',
    'PreSubmitted': 'pending',
    'Submitted': 'pending',
    'Filled': 'filled',
    'Cancelled': 'cancelled',
    'Inactive': 'rejected',
    'ApiCancelled': 'cancelled',
    'PendingCancel': 'cancelled',
}


def _map_status(ibkr_status: str) -> str:
    return _STATUS_MAP.get(ibkr_status, 'error')


class IBKRBrokerAdapter(AbstractBroker):
    """Interactive Brokers adapter via IB Gateway / TWS."""

    def __init__(
        self,
        config: Optional[BrokerConfig] = None,
        conn: Optional[IBKRConnectionManager] = None,
    ) -> None:
        if config is None:
            env = os.environ.get('IBKR_ENVIRONMENT', 'paper')
            config = BrokerConfig(broker_id='ibkr', environment=env)
        super().__init__(config)
        self._conn = conn or IBKRConnectionManager()
        self._market_hours = USEquityHours()

    @property
    def conn(self) -> IBKRConnectionManager:
        return self._conn

    # -- Connection / Account ------------------------------------------------

    def is_configured(self) -> bool:
        host = os.environ.get('IBKR_HOST', '')
        port = os.environ.get('IBKR_PORT', '')
        return bool(host and port)

    async def get_account(self) -> Optional[Dict]:
        if not await self._conn.ensure_connected():
            return None
        try:
            summary = await self._conn.ib.accountSummaryAsync()
            result: Dict[str, str] = {}
            for item in summary:
                result[item.tag] = item.value

            return {
                'equity': float(result.get('NetLiquidation', 0)),
                'buying_power': float(result.get('BuyingPower', 0)),
                'cash': float(result.get('TotalCashValue', 0)),
                'portfolio_value': float(result.get('GrossPositionValue', 0)),
                'unrealized_pnl': float(result.get('UnrealizedPnL', 0)),
                'broker_id': 'ibkr',
                'raw': result,
            }
        except Exception as e:
            logger.error(f"IBKR get_account error: {e}")
            return None

    async def get_positions(self) -> List[Dict]:
        if not await self._conn.ensure_connected():
            return []
        try:
            positions = self._conn.ib.positions()
            result = []
            for pos in positions:
                qty = pos.position
                if qty == 0:
                    continue
                result.append({
                    'symbol': pos.contract.symbol,
                    'qty': abs(float(qty)),
                    'avg_entry_price': float(pos.avgCost),
                    'side': 'long' if qty > 0 else 'short',
                    'broker_id': 'ibkr',
                    'con_id': pos.contract.conId,
                })
            return result
        except Exception as e:
            logger.error(f"IBKR get_positions error: {e}")
            return []

    # -- Order Execution -----------------------------------------------------

    async def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        timeout_sec: float = 30.0,
    ) -> OrderResult:
        if not await self._conn.ensure_connected():
            return OrderResult(symbol=symbol, side=side, status='error',
                               error='IBKR not connected')

        contract = await self._conn.qualify_contract(symbol)
        if not contract:
            return OrderResult(symbol=symbol, side=side, status='error',
                               error=f'Cannot qualify contract for {symbol}')

        # Map side to IBKR action
        action = 'BUY' if side in ('buy', 'long') else 'SELL'
        int_qty = int(abs(qty))

        if order_type == 'limit' and limit_price:
            order = LimitOrder(action, int_qty, limit_price, tif='DAY')
        else:
            order = MarketOrder(action, int_qty, tif='DAY')

        try:
            trade = self._conn.ib.placeOrder(contract, order)
            logger.info(
                f"IBKR order placed: {action} {int_qty} {symbol} "
                f"type={order_type} orderId={trade.order.orderId}"
            )
        except Exception as e:
            logger.error(f"IBKR placeOrder error: {e}")
            return OrderResult(symbol=symbol, side=side, status='error',
                               error=str(e))

        # Poll for fill
        return await self._wait_for_fill(trade, symbol, side, timeout_sec)

    async def _wait_for_fill(
        self, trade: Trade, symbol: str, side: str, timeout_sec: float,
    ) -> OrderResult:
        """Poll trade status until filled, rejected, or timeout."""
        deadline = time.monotonic() + timeout_sec
        poll_interval = 0.3

        while time.monotonic() < deadline:
            status = trade.orderStatus.status

            if status == 'Filled':
                return OrderResult(
                    order_id=str(trade.order.orderId),
                    status='filled',
                    filled_qty=float(trade.orderStatus.filled),
                    filled_avg_price=float(trade.orderStatus.avgFillPrice),
                    symbol=symbol,
                    side=side,
                )

            if status in ('Cancelled', 'Inactive', 'ApiCancelled'):
                return OrderResult(
                    order_id=str(trade.order.orderId),
                    status=_map_status(status),
                    symbol=symbol,
                    side=side,
                    error=f'Order {status}: {trade.log[-1].message if trade.log else ""}',
                )

            await asyncio.sleep(poll_interval)

        # Timeout
        final_status = trade.orderStatus.status
        filled = float(trade.orderStatus.filled)
        if filled > 0:
            return OrderResult(
                order_id=str(trade.order.orderId),
                status='partially_filled',
                filled_qty=filled,
                filled_avg_price=float(trade.orderStatus.avgFillPrice),
                symbol=symbol,
                side=side,
                error=f'Partial fill after {timeout_sec}s ({final_status})',
            )

        return OrderResult(
            order_id=str(trade.order.orderId),
            status='pending',
            symbol=symbol,
            side=side,
            error=f'Still {final_status} after {timeout_sec}s',
        )

    async def place_stop_order(
        self,
        symbol: str,
        qty: float,
        stop_price: float,
        limit_offset_pct: float = 0.003,
        direction: str = 'short',
    ) -> Dict:
        if not await self._conn.ensure_connected():
            return {'error': 'IBKR not connected'}

        contract = await self._conn.qualify_contract(symbol)
        if not contract:
            return {'error': f'Cannot qualify contract for {symbol}'}

        int_qty = int(abs(qty))
        if direction == 'long':
            action = 'SELL'
            limit_price = round(stop_price * (1 - limit_offset_pct), 2)
        else:
            action = 'BUY'
            limit_price = round(stop_price * (1 + limit_offset_pct), 2)

        # StopLimitOrder(action, qty, lmtPrice, stopPrice)
        order = StopLimitOrder(
            action, int_qty, limit_price, stop_price, tif='DAY',
        )

        try:
            trade = self._conn.ib.placeOrder(contract, order)
            order_id = str(trade.order.orderId)
            logger.info(
                f"IBKR stop placed: {action} {int_qty} {symbol} "
                f"stop=${stop_price:.2f} limit=${limit_price:.2f} id={order_id}"
            )
            return {
                'id': order_id,
                'status': trade.orderStatus.status,
                'symbol': symbol,
                'stop_price': stop_price,
                'limit_price': limit_price,
            }
        except Exception as e:
            logger.error(f"IBKR stop order error: {e}")
            return {'error': str(e)}

    async def cancel_order(self, order_id: str) -> bool:
        if not await self._conn.ensure_connected():
            return False
        try:
            # Find the trade by order ID
            for trade in self._conn.ib.openTrades():
                if str(trade.order.orderId) == order_id:
                    self._conn.ib.cancelOrder(trade.order)
                    logger.info(f"IBKR cancel order: {order_id}")
                    return True
            logger.warning(f"IBKR cancel order: {order_id} not found in open trades")
            return False
        except Exception as e:
            logger.error(f"IBKR cancel order error: {e}")
            return False

    async def get_order(self, order_id: str) -> Optional[Dict]:
        if not await self._conn.ensure_connected():
            return None
        try:
            for trade in self._conn.ib.trades():
                if str(trade.order.orderId) == order_id:
                    os = trade.orderStatus
                    return {
                        'id': order_id,
                        'status': _map_status(os.status),
                        'ibkr_status': os.status,
                        'filled_qty': float(os.filled),
                        'avg_fill_price': float(os.avgFillPrice),
                        'remaining': float(os.remaining),
                        'symbol': trade.contract.symbol,
                    }
            return None
        except Exception as e:
            logger.error(f"IBKR get_order error: {e}")
            return None

    async def replace_stop_order(
        self,
        old_order_id: str,
        symbol: str,
        qty: float,
        new_stop_price: float,
        limit_offset_pct: float = 0.003,
    ) -> Dict:
        if old_order_id:
            cancelled = await self.cancel_order(old_order_id)
            if not cancelled:
                # Verify the order is actually gone before placing a new one
                existing = await self.get_order(old_order_id)
                if existing and existing.get('status') not in ('cancelled', 'filled'):
                    return {'error': f'Cannot cancel order {old_order_id} before replacing'}
            await asyncio.sleep(0.5)  # Let cancellation propagate
        return await self.place_stop_order(
            symbol, qty, new_stop_price, limit_offset_pct,
        )

    # -- Market Data ---------------------------------------------------------

    def create_tick_streamer(
        self,
        symbols: List[str],
        on_tick: Optional[Callable] = None,
    ) -> TickStreamer:
        return IBKRTickStreamer(symbols, self._conn, on_tick=on_tick)

    async def fetch_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Optional[pd.DataFrame]:
        if not await self._conn.ensure_connected():
            return None

        contract = await self._conn.qualify_contract(symbol)
        if not contract:
            return None

        # Map interval to IBKR format
        bar_size = _interval_to_ibkr(interval)
        duration = _date_range_to_duration(start_date, end_date)

        try:
            bars = await self._conn.ib.reqHistoricalDataAsync(
                contract,
                endDateTime=end_date.replace('-', '') + ' 23:59:59',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=True,
            )
            if not bars:
                return None

            data = [{
                'date': b.date,
                'open': b.open,
                'high': b.high,
                'low': b.low,
                'close': b.close,
                'volume': b.volume,
            } for b in bars]
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logger.error(f"IBKR fetch_bars {symbol} error: {e}")
            return None

    async def fetch_bars_multi(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = '1Day',
    ) -> Dict[str, pd.DataFrame]:
        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = await self.fetch_bars(sym, start_date, end_date, interval)
            if df is not None:
                result[sym] = df
            # Respect IBKR pacing: max 60 historical requests per 10 min
            await asyncio.sleep(0.5)
        return result

    # -- Symbol Info ----------------------------------------------------------

    async def check_tradeable(self, symbol: str) -> Tuple[bool, bool]:
        if not await self._conn.ensure_connected():
            return False, False

        contract = await self._conn.qualify_contract(symbol)
        if not contract:
            return False, False

        # Check shortability via contract details (no market data subscription needed)
        try:
            details = await self._conn.ib.reqContractDetailsAsync(contract)
            if not details:
                return False, False
            # If contract qualifies, it's tradeable. Use streaming tick 236
            # briefly for shortable shares info.
            ticker = self._conn.ib.reqMktData(
                contract, genericTickList='236', snapshot=False,
            )
            await asyncio.sleep(2)
            shortable = ticker.shortableShares
            self._conn.ib.cancelMktData(ticker.contract)

            if shortable is None or shortable <= 0:
                return True, False  # tradeable but not shortable
            return True, shortable > 1_000_000
        except Exception as e:
            logger.warning(f"IBKR check_tradeable {symbol} error: {e}")
            return False, False

    async def get_tradeable_symbols(
        self,
        min_price: float = 1.0,
        max_price: float = 0.0,
    ) -> List[str]:
        # IBKR has no symbol listing API — use DB-sourced universe
        return []

    # -- Snapshot Prices -----------------------------------------------------

    async def get_snapshot_prices(self, symbols: List[str]) -> Dict[str, float]:
        if not await self._conn.ensure_connected():
            return {}

        contracts = await self._conn.qualify_contracts_batch(symbols)
        if not contracts:
            return {}

        # Request delayed data if real-time not available
        self._conn.ib.reqMarketDataType(3)  # 3 = delayed if no real-time sub

        prices: Dict[str, float] = {}
        tickers = []
        try:
            for sym, contract in contracts.items():
                ticker = self._conn.ib.reqMktData(
                    contract, genericTickList='', snapshot=True,
                )
                tickers.append((sym, contract, ticker))

            # Wait for snapshots to populate
            await asyncio.sleep(3)

            for sym, contract, ticker in tickers:
                price = ticker.marketPrice()
                if price != price or price <= 0:  # NaN check
                    price = ticker.last if ticker.last > 0 else ticker.close
                if price != price or price <= 0:
                    # Try delayed data fields
                    price = ticker.delayedLast if hasattr(ticker, 'delayedLast') and ticker.delayedLast > 0 else 0
                if price > 0 and price == price:
                    prices[sym] = float(price)
                self._conn.ib.cancelMktData(contract)
        finally:
            # Always reset to live data type for streaming
            self._conn.ib.reqMarketDataType(1)

        return prices

    # -- Market Hours --------------------------------------------------------

    def get_market_hours(self) -> MarketHoursPolicy:
        return self._market_hours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interval_to_ibkr(interval: str) -> str:
    """Convert common interval strings to IBKR bar size format."""
    mapping = {
        '1min': '1 min',
        '1Min': '1 min',
        '5min': '5 mins',
        '5Min': '5 mins',
        '15min': '15 mins',
        '15Min': '15 mins',
        '1hour': '1 hour',
        '1Hour': '1 hour',
        '1day': '1 day',
        '1Day': '1 day',
    }
    return mapping.get(interval, '1 day')


def _date_range_to_duration(start_date: str, end_date: str) -> str:
    """Estimate IBKR duration string from date range."""
    from datetime import date
    try:
        d1 = date.fromisoformat(start_date)
        d2 = date.fromisoformat(end_date)
        days = (d2 - d1).days + 1
        if days <= 5:
            return f'{days} D'
        elif days <= 30:
            return f'{days} D'
        elif days <= 365:
            months = max(1, days // 30)
            return f'{months} M'
        else:
            years = max(1, days // 365)
            return f'{years} Y'
    except Exception:
        return '6 M'
