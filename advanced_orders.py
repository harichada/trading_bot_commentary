#!/usr/bin/env python3
"""
Advanced Order Management System
Supports trailing stops, DCA, bracket orders, and more
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import uuid
import logging
import numpy as np

logger = logging.getLogger('AdvancedOrders')

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    TRAILING_STOP_LIMIT = "TRAILING_STOP_LIMIT"
    OCO = "OCO"  # One Cancels Other
    BRACKET = "BRACKET"
    DCA = "DCA"  # Dollar Cost Averaging
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"  # Time Weighted Average Price
    VWAP = "VWAP"  # Volume Weighted Average Price

class OrderStatus(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    """Base order class"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    quantity: float = 0
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Price fields
    price: Optional[float] = None  # Limit price
    stop_price: Optional[float] = None  # Stop price
    average_fill_price: Optional[float] = None
    
    # Execution details
    filled_quantity: float = 0
    remaining_quantity: float = 0
    fills: List[Dict[str, Any]] = field(default_factory=list)
    
    # Time in force
    time_in_force: str = "DAY"  # DAY, GTC, IOC, FOK
    expire_time: Optional[datetime] = None
    
    # Advanced fields
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.remaining_quantity = self.quantity

@dataclass
class TrailingStopOrder(Order):
    """Trailing stop order with dynamic adjustment"""
    trail_amount: Optional[float] = None  # Dollar amount
    trail_percent: Optional[float] = None  # Percentage
    high_water_mark: Optional[float] = None  # Best price seen
    
    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.TRAILING_STOP
        
        if self.trail_amount is None and self.trail_percent is None:
            raise ValueError("Either trail_amount or trail_percent must be specified")
    
    def update_stop_price(self, current_price: float):
        """Update stop price based on current market price"""
        if self.side == OrderSide.SELL:
            # For sell orders, track highest price
            if self.high_water_mark is None or current_price > self.high_water_mark:
                self.high_water_mark = current_price
                
                if self.trail_amount:
                    new_stop = self.high_water_mark - self.trail_amount
                else:
                    new_stop = self.high_water_mark * (1 - self.trail_percent)
                
                if self.stop_price is None or new_stop > self.stop_price:
                    self.stop_price = new_stop
                    self.updated_at = datetime.now()
                    logger.info(f"Trailing stop updated: {self.symbol} stop at ${self.stop_price:.2f}")
                    
        else:
            # For buy orders, track lowest price
            if self.high_water_mark is None or current_price < self.high_water_mark:
                self.high_water_mark = current_price
                
                if self.trail_amount:
                    new_stop = self.high_water_mark + self.trail_amount
                else:
                    new_stop = self.high_water_mark * (1 + self.trail_percent)
                
                if self.stop_price is None or new_stop < self.stop_price:
                    self.stop_price = new_stop
                    self.updated_at = datetime.now()

@dataclass
class BracketOrder(Order):
    """Bracket order with profit target and stop loss"""
    entry_order: Order = None
    take_profit_order: Order = None
    stop_loss_order: Order = None
    
    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.BRACKET
        
        if self.entry_order:
            self.child_order_ids = [
                self.entry_order.order_id,
                self.take_profit_order.order_id if self.take_profit_order else None,
                self.stop_loss_order.order_id if self.stop_loss_order else None
            ]
            self.child_order_ids = [oid for oid in self.child_order_ids if oid]

@dataclass
class DCAOrder(Order):
    """Dollar Cost Averaging order"""
    total_amount: float = 0  # Total amount to invest
    interval: timedelta = timedelta(hours=1)  # Time between orders
    num_orders: int = 10  # Number of orders to split into
    orders_placed: int = 0
    next_order_time: datetime = field(default_factory=datetime.now)
    child_orders: List[Order] = field(default_factory=list)
    
    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.DCA
        self.quantity = self.total_amount  # Store total amount
        
    def get_next_order_quantity(self, current_price: float) -> float:
        """Calculate quantity for next DCA order"""
        remaining_orders = self.num_orders - self.orders_placed
        if remaining_orders <= 0:
            return 0
        
        remaining_amount = self.total_amount - sum(o.quantity * o.price for o in self.child_orders)
        order_amount = remaining_amount / remaining_orders
        
        return order_amount / current_price
    
    def should_place_next_order(self) -> bool:
        """Check if it's time to place the next order"""
        return (datetime.now() >= self.next_order_time and 
                self.orders_placed < self.num_orders)

@dataclass
class IcebergOrder(Order):
    """Iceberg order - only shows part of total quantity"""
    total_quantity: float = 0
    visible_quantity: float = 0  # Quantity shown in order book
    executed_quantity: float = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.ICEBERG
        self.quantity = self.visible_quantity  # Only show visible part
        
    def get_next_slice(self) -> float:
        """Get next slice to show after current is filled"""
        remaining = self.total_quantity - self.executed_quantity
        return min(self.visible_quantity, remaining)

@dataclass
class TWAPOrder(Order):
    """Time Weighted Average Price order"""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = None
    interval: timedelta = timedelta(minutes=5)
    total_quantity: float = 0
    slices_completed: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        self.order_type = OrderType.TWAP
        
        if self.end_time is None:
            self.end_time = self.start_time + timedelta(hours=1)
    
    def get_slice_quantity(self) -> float:
        """Calculate quantity for each time slice"""
        total_duration = (self.end_time - self.start_time).total_seconds()
        num_slices = int(total_duration / self.interval.total_seconds())
        
        return self.total_quantity / num_slices
    
    def get_next_slice_time(self) -> datetime:
        """Get time for next slice"""
        return self.start_time + (self.interval * (self.slices_completed + 1))

class OrderManager:
    """Manages advanced order types and execution"""
    
    def __init__(self, execute_callback: Callable):
        self.orders: Dict[str, Order] = {}
        self.active_orders: List[str] = []
        self.execute_callback = execute_callback  # Function to execute orders
        self.market_data_callbacks: Dict[str, Callable] = {}
        
    def place_order(self, order: Order) -> str:
        """Place an order"""
        order_id = order.order_id
        self.orders[order_id] = order
        self.active_orders.append(order_id)
        
        logger.info(f"Order placed: {order.order_type.value} {order.side.value} "
                   f"{order.quantity} {order.symbol} @ "
                   f"{'Market' if order.price is None else f'${order.price:.2f}'}")
        
        # Handle different order types
        if order.order_type == OrderType.BRACKET:
            self._handle_bracket_order(order)
        elif order.order_type == OrderType.DCA:
            self._handle_dca_order(order)
        elif order.order_type == OrderType.TRAILING_STOP:
            self._setup_trailing_stop(order)
        elif order.order_type == OrderType.TWAP:
            self._handle_twap_order(order)
        elif order.order_type == OrderType.ICEBERG:
            self._handle_iceberg_order(order)
        else:
            # Simple order types
            self._execute_simple_order(order)
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.updated_at = datetime.now()
        
        # Cancel child orders
        for child_id in order.child_order_ids:
            self.cancel_order(child_id)
        
        # Remove from active orders
        if order_id in self.active_orders:
            self.active_orders.remove(order_id)
        
        logger.info(f"Order cancelled: {order_id}")
        return True
    
    def update_order(self, order_id: str, updates: Dict[str, Any]) -> bool:
        """Update an order (if not filled)"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        # Update allowed fields
        allowed_updates = ['price', 'quantity', 'stop_price']
        for field, value in updates.items():
            if field in allowed_updates and hasattr(order, field):
                setattr(order, field, value)
        
        order.updated_at = datetime.now()
        logger.info(f"Order updated: {order_id}")
        return True
    
    def _handle_bracket_order(self, bracket: BracketOrder):
        """Handle bracket order execution"""
        # First place the entry order
        entry_order = bracket.entry_order
        entry_order.parent_order_id = bracket.order_id
        
        def on_entry_fill(filled_order):
            # Once entry is filled, place the bracket orders
            if bracket.take_profit_order:
                tp_order = bracket.take_profit_order
                tp_order.parent_order_id = bracket.order_id
                tp_order.quantity = filled_order.filled_quantity
                self.place_order(tp_order)
            
            if bracket.stop_loss_order:
                sl_order = bracket.stop_loss_order
                sl_order.parent_order_id = bracket.order_id
                sl_order.quantity = filled_order.filled_quantity
                self.place_order(sl_order)
            
            # Make them OCO if both exist
            if bracket.take_profit_order and bracket.stop_loss_order:
                self._link_oco_orders(
                    bracket.take_profit_order.order_id,
                    bracket.stop_loss_order.order_id
                )
        
        entry_order.metadata['on_fill_callback'] = on_entry_fill
        self.place_order(entry_order)
    
    def _handle_dca_order(self, dca: DCAOrder):
        """Handle DCA order execution"""
        def place_next_dca_slice():
            if not dca.should_place_next_order():
                return
            
            # Get current price (would come from market data)
            current_price = self._get_current_price(dca.symbol)
            quantity = dca.get_next_order_quantity(current_price)
            
            if quantity > 0:
                # Create child order
                child_order = Order(
                    symbol=dca.symbol,
                    side=dca.side,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    parent_order_id=dca.order_id
                )
                
                dca.child_orders.append(child_order)
                dca.orders_placed += 1
                dca.next_order_time = datetime.now() + dca.interval
                
                self.place_order(child_order)
                
                # Schedule next order
                if dca.orders_placed < dca.num_orders:
                    self._schedule_order(dca.next_order_time, place_next_dca_slice)
        
        # Place first order immediately
        place_next_dca_slice()
    
    def _handle_twap_order(self, twap: TWAPOrder):
        """Handle TWAP order execution"""
        def place_next_twap_slice():
            if datetime.now() > twap.end_time:
                return
            
            slice_quantity = twap.get_slice_quantity()
            
            # Create child order
            child_order = Order(
                symbol=twap.symbol,
                side=twap.side,
                quantity=slice_quantity,
                order_type=OrderType.MARKET,
                parent_order_id=twap.order_id
            )
            
            twap.slices_completed += 1
            self.place_order(child_order)
            
            # Schedule next slice
            next_time = twap.get_next_slice_time()
            if next_time < twap.end_time:
                self._schedule_order(next_time, place_next_twap_slice)
        
        # Place first slice
        place_next_twap_slice()
    
    def _handle_iceberg_order(self, iceberg: IcebergOrder):
        """Handle iceberg order execution"""
        def on_slice_fill(filled_order):
            iceberg.executed_quantity += filled_order.filled_quantity
            
            # Place next slice if more remaining
            next_slice = iceberg.get_next_slice()
            if next_slice > 0:
                child_order = Order(
                    symbol=iceberg.symbol,
                    side=iceberg.side,
                    quantity=next_slice,
                    order_type=OrderType.LIMIT,
                    price=iceberg.price,
                    parent_order_id=iceberg.order_id
                )
                child_order.metadata['on_fill_callback'] = on_slice_fill
                self.place_order(child_order)
        
        # Place first visible slice
        first_order = Order(
            symbol=iceberg.symbol,
            side=iceberg.side,
            quantity=iceberg.visible_quantity,
            order_type=OrderType.LIMIT,
            price=iceberg.price,
            parent_order_id=iceberg.order_id
        )
        first_order.metadata['on_fill_callback'] = on_slice_fill
        self.place_order(first_order)
    
    def _setup_trailing_stop(self, trailing_stop: TrailingStopOrder):
        """Setup trailing stop monitoring"""
        def update_trailing_stop(symbol: str, price: float):
            if trailing_stop.status == OrderStatus.PENDING:
                trailing_stop.update_stop_price(price)
                
                # Check if stop is triggered
                if trailing_stop.side == OrderSide.SELL and price <= trailing_stop.stop_price:
                    self._execute_simple_order(trailing_stop)
                elif trailing_stop.side == OrderSide.BUY and price >= trailing_stop.stop_price:
                    self._execute_simple_order(trailing_stop)
        
        # Register for price updates
        self.market_data_callbacks[trailing_stop.symbol] = update_trailing_stop
    
    def _execute_simple_order(self, order: Order):
        """Execute a simple order"""
        order.status = OrderStatus.SUBMITTED
        
        # Call the execution callback (broker API)
        result = self.execute_callback(order)
        
        if result['success']:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.remaining_quantity = 0
            order.average_fill_price = result.get('fill_price', order.price)
            
            # Handle callbacks
            if 'on_fill_callback' in order.metadata:
                order.metadata['on_fill_callback'](order)
        else:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {result.get('reason', 'Unknown')}")
    
    def _link_oco_orders(self, order_id1: str, order_id2: str):
        """Link two orders as OCO"""
        if order_id1 in self.orders and order_id2 in self.orders:
            order1 = self.orders[order_id1]
            order2 = self.orders[order_id2]
            
            # Set up callbacks to cancel the other when one fills
            def cancel_other(filled_order):
                other_id = order_id2 if filled_order.order_id == order_id1 else order_id1
                self.cancel_order(other_id)
            
            order1.metadata['on_fill_callback'] = cancel_other
            order2.metadata['on_fill_callback'] = cancel_other
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current market price (placeholder)"""
        # In real implementation, this would fetch from market data
        return 100.0
    
    def _schedule_order(self, when: datetime, callback: Callable):
        """Schedule an order for future execution (placeholder)"""
        # In real implementation, this would use a scheduler
        pass
    
    def update_market_data(self, symbol: str, price: float):
        """Update market data and trigger callbacks"""
        if symbol in self.market_data_callbacks:
            self.market_data_callbacks[symbol](symbol, price)
    
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get order status"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders"""
        return [self.orders[oid] for oid in self.active_orders if oid in self.orders]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [o for o in self.orders.values() if o.symbol == symbol]

# Helper functions for creating orders
def create_bracket_order(symbol: str, side: OrderSide, quantity: float,
                        entry_price: float, take_profit: float, stop_loss: float) -> BracketOrder:
    """Create a bracket order"""
    entry = Order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=entry_price
    )
    
    tp_side = OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY
    
    take_profit_order = Order(
        symbol=symbol,
        side=tp_side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        price=take_profit
    )
    
    stop_loss_order = Order(
        symbol=symbol,
        side=tp_side,
        quantity=quantity,
        order_type=OrderType.STOP,
        stop_price=stop_loss
    )
    
    return BracketOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_order=entry,
        take_profit_order=take_profit_order,
        stop_loss_order=stop_loss_order
    )

def create_trailing_stop(symbol: str, side: OrderSide, quantity: float,
                        trail_amount: float = None, trail_percent: float = None) -> TrailingStopOrder:
    """Create a trailing stop order"""
    return TrailingStopOrder(
        symbol=symbol,
        side=side,
        quantity=quantity,
        trail_amount=trail_amount,
        trail_percent=trail_percent
    )

def create_dca_order(symbol: str, side: OrderSide, total_amount: float,
                    num_orders: int = 10, interval_hours: int = 1) -> DCAOrder:
    """Create a DCA order"""
    return DCAOrder(
        symbol=symbol,
        side=side,
        total_amount=total_amount,
        num_orders=num_orders,
        interval=timedelta(hours=interval_hours)
    )