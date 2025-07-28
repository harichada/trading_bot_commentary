#!/usr/bin/env python3
"""
Paper Trading System
Risk-free trading simulation with realistic execution
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
import asyncio
from abc import ABC, abstractmethod

# Import components from other modules
from advanced_orders import Order, OrderType, OrderSide, OrderStatus
from performance_analytics import PerformanceAnalyzer, PerformanceTracker

logger = logging.getLogger('PaperTrading')

class ExecutionModel(Enum):
    OPTIMISTIC = "OPTIMISTIC"  # Best case execution
    REALISTIC = "REALISTIC"    # With slippage and spread
    PESSIMISTIC = "PESSIMISTIC"  # Worst case execution

@dataclass
class PaperPosition:
    """Represents a paper trading position"""
    symbol: str
    quantity: float
    side: str  # 'long' or 'short'
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    highest_price: float = 0  # For trailing stops
    lowest_price: float = float('inf')  # For short trailing stops
    unrealized_pnl: float = 0
    commission_paid: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        
        # Update highest/lowest for trailing stops
        if self.side == 'long':
            self.highest_price = max(self.highest_price, new_price)
            self.unrealized_pnl = (new_price - self.entry_price) * self.quantity
        else:  # short
            self.lowest_price = min(self.lowest_price, new_price)
            self.unrealized_pnl = (self.entry_price - new_price) * abs(self.quantity)
        
        # Update trailing stop if applicable
        if self.trailing_stop:
            self._update_trailing_stop()
    
    def _update_trailing_stop(self):
        """Update trailing stop level"""
        if self.side == 'long' and self.trailing_stop:
            new_stop = self.highest_price - self.trailing_stop
            if self.stop_loss is None or new_stop > self.stop_loss:
                self.stop_loss = new_stop
        elif self.side == 'short' and self.trailing_stop:
            new_stop = self.lowest_price + self.trailing_stop
            if self.stop_loss is None or new_stop < self.stop_loss:
                self.stop_loss = new_stop

@dataclass
class PaperOrder:
    """Paper trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    commission: float = 0
    slippage: float = 0
    rejection_reason: Optional[str] = None

@dataclass
class PaperAccount:
    """Paper trading account"""
    initial_balance: float
    current_balance: float
    buying_power: float
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: Dict[str, PaperOrder] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_history: List[Tuple[datetime, float]] = field(default_factory=list)
    
    @property
    def total_equity(self) -> float:
        """Calculate total account equity"""
        positions_value = sum(
            pos.quantity * pos.current_price if pos.side == 'long'
            else pos.quantity * (2 * pos.entry_price - pos.current_price)
            for pos in self.positions.values()
        )
        return self.current_balance + positions_value
    
    @property
    def margin_used(self) -> float:
        """Calculate margin used (for short positions)"""
        return sum(
            abs(pos.quantity) * pos.current_price * 0.5  # 50% margin for shorts
            for pos in self.positions.values()
            if pos.side == 'short'
        )
    
    @property
    def available_margin(self) -> float:
        """Calculate available margin"""
        return max(0, self.buying_power - self.margin_used)

class MarketSimulator:
    """Simulates market conditions for paper trading"""
    
    def __init__(self, execution_model: ExecutionModel = ExecutionModel.REALISTIC):
        self.execution_model = execution_model
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.current_prices: Dict[str, float] = {}
        self.bid_ask_spreads: Dict[str, float] = {}
        
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """Update market data for symbol"""
        self.market_data[symbol] = data
        if len(data) > 0:
            self.current_prices[symbol] = data.iloc[-1]['close']
            
            # Estimate spread based on volatility
            volatility = data['close'].pct_change().std()
            self.bid_ask_spreads[symbol] = max(0.01, volatility * 2)  # 2x volatility as spread
    
    def get_execution_price(self, symbol: str, side: OrderSide, 
                          order_type: OrderType, limit_price: Optional[float] = None) -> Tuple[float, float]:
        """Get execution price with slippage and spread"""
        if symbol not in self.current_prices:
            return 0, 0
        
        base_price = self.current_prices[symbol]
        spread = self.bid_ask_spreads.get(symbol, 0.01)
        
        # Calculate bid/ask
        bid = base_price - spread / 2
        ask = base_price + spread / 2
        
        # Base execution price
        if side == OrderSide.BUY:
            exec_price = ask
        else:
            exec_price = bid
        
        # Apply execution model
        if self.execution_model == ExecutionModel.OPTIMISTIC:
            # Best case - might get price improvement
            slippage = -spread * 0.1  # 10% price improvement
        elif self.execution_model == ExecutionModel.REALISTIC:
            # Realistic - some slippage
            slippage = spread * 0.2  # 20% additional slippage
        else:  # PESSIMISTIC
            # Worst case - significant slippage
            slippage = spread * 0.5  # 50% additional slippage
        
        # Apply slippage based on side
        if side == OrderSide.BUY:
            exec_price += abs(slippage)
        else:
            exec_price -= abs(slippage)
        
        # Handle limit orders
        if order_type == OrderType.LIMIT and limit_price:
            if side == OrderSide.BUY and exec_price > limit_price:
                return 0, 0  # Cannot execute
            elif side == OrderSide.SELL and exec_price < limit_price:
                return 0, 0  # Cannot execute
            exec_price = limit_price
        
        return exec_price, abs(slippage)
    
    def check_stop_trigger(self, symbol: str, stop_price: float, side: OrderSide) -> bool:
        """Check if stop price has been triggered"""
        if symbol not in self.current_prices:
            return False
        
        current = self.current_prices[symbol]
        
        if side == OrderSide.BUY:
            return current >= stop_price
        else:
            return current <= stop_price
    
    def estimate_market_impact(self, symbol: str, quantity: float) -> float:
        """Estimate market impact of large orders"""
        if symbol not in self.market_data or len(self.market_data[symbol]) == 0:
            return 0
        
        # Get average volume
        avg_volume = self.market_data[symbol]['volume'].rolling(20).mean().iloc[-1]
        
        if avg_volume == 0:
            return 0
        
        # Calculate order size relative to average volume
        order_pct = quantity / avg_volume
        
        # Estimate impact (simplified square-root model)
        impact = 0.1 * np.sqrt(order_pct * 100)  # 0.1% per 1% of volume
        
        return min(impact, 0.02)  # Cap at 2%

class PaperTradingEngine:
    """Main paper trading engine"""
    
    def __init__(self, initial_balance: float = 100000, 
                 execution_model: ExecutionModel = ExecutionModel.REALISTIC):
        self.account = PaperAccount(
            initial_balance=initial_balance,
            current_balance=initial_balance,
            buying_power=initial_balance
        )
        
        self.market_simulator = MarketSimulator(execution_model)
        self.performance_analyzer = PerformanceAnalyzer()
        self.performance_tracker = PerformanceTracker(self.performance_analyzer)
        
        # Configuration
        self.commission_rate = 0.001  # 0.1% per trade
        self.min_commission = 1.0  # $1 minimum
        self.allow_shorting = True
        self.use_margin = True
        self.margin_requirement = 0.5  # 50% for shorts
        
        # Order management
        self.pending_orders: Dict[str, PaperOrder] = {}
        self.order_counter = 0
        
        # State persistence
        self.state_file = Path("paper_trading_state.json")
        
    def place_order(self, order: Order) -> str:
        """Place a paper trading order"""
        # Generate order ID
        self.order_counter += 1
        order_id = f"PT_{self.order_counter:06d}"
        
        # Validate order
        validation_result = self._validate_order(order)
        if not validation_result['valid']:
            logger.warning(f"Order rejected: {validation_result['reason']}")
            return ""
        
        # Create paper order
        paper_order = PaperOrder(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            order_type=order.order_type,
            price=order.price,
            stop_price=order.stop_price,
            status=OrderStatus.PENDING,
            created_at=datetime.now()
        )
        
        # Add to pending orders
        self.pending_orders[order_id] = paper_order
        self.account.orders[order_id] = paper_order
        
        # Try immediate execution for market orders
        if order.order_type == OrderType.MARKET:
            self._execute_order(paper_order)
        
        logger.info(f"Paper order placed: {order_id} - {order.side.value} {order.quantity} {order.symbol}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a paper order"""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                del self.pending_orders[order_id]
                logger.info(f"Paper order cancelled: {order_id}")
                return True
        return False
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """Update market data and check orders/positions"""
        self.market_simulator.update_market_data(symbol, data)
        
        # Check pending orders
        self._check_pending_orders(symbol)
        
        # Update positions
        self._update_positions(symbol)
        
        # Check stop losses and take profits
        self._check_exits(symbol)
        
        # Update equity history
        self.account.equity_history.append((datetime.now(), self.account.total_equity))
        
        # Update performance analyzer
        self.performance_analyzer.update_equity(datetime.now(), self.account.total_equity)
    
    def _validate_order(self, order: Order) -> Dict[str, Any]:
        """Validate order before execution"""
        # Check symbol
        if not order.symbol:
            return {'valid': False, 'reason': 'Invalid symbol'}
        
        # Check quantity
        if order.quantity <= 0:
            return {'valid': False, 'reason': 'Invalid quantity'}
        
        # Check buying power for buys
        if order.side == OrderSide.BUY:
            est_cost = order.quantity * (order.price or self.market_simulator.current_prices.get(order.symbol, 0))
            if est_cost > self.account.buying_power:
                return {'valid': False, 'reason': 'Insufficient buying power'}
        
        # Check position for sells (if not shorting)
        if order.side == OrderSide.SELL and not self.allow_shorting:
            if order.symbol not in self.account.positions:
                return {'valid': False, 'reason': 'No position to sell'}
            if order.quantity > self.account.positions[order.symbol].quantity:
                return {'valid': False, 'reason': 'Insufficient position size'}
        
        return {'valid': True}
    
    def _execute_order(self, order: PaperOrder):
        """Execute a paper order"""
        # Get execution price
        exec_price, slippage = self.market_simulator.get_execution_price(
            order.symbol, order.side, order.order_type, order.price
        )
        
        if exec_price == 0:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Cannot execute at limit price"
            return
        
        # Calculate commission
        commission = max(self.min_commission, order.quantity * exec_price * self.commission_rate)
        
        # Execute based on side
        if order.side == OrderSide.BUY:
            self._execute_buy(order, exec_price, commission, slippage)
        else:
            self._execute_sell(order, exec_price, commission, slippage)
        
        # Update order
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        order.filled_price = exec_price
        order.commission = commission
        order.slippage = slippage
        
        # Remove from pending
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
        
        logger.info(f"Paper order filled: {order.order_id} @ ${exec_price:.2f}")
    
    def _execute_buy(self, order: PaperOrder, price: float, commission: float, slippage: float):
        """Execute buy order"""
        total_cost = order.quantity * price + commission
        
        # Check buying power
        if total_cost > self.account.buying_power:
            order.status = OrderStatus.REJECTED
            order.rejection_reason = "Insufficient buying power"
            return
        
        # Update account
        self.account.current_balance -= total_cost
        self.account.buying_power -= total_cost
        
        # Create or update position
        if order.symbol in self.account.positions:
            # Average into existing position
            pos = self.account.positions[order.symbol]
            total_quantity = pos.quantity + order.quantity
            avg_price = ((pos.quantity * pos.entry_price) + (order.quantity * price)) / total_quantity
            
            pos.quantity = total_quantity
            pos.entry_price = avg_price
            pos.commission_paid += commission
        else:
            # New position
            position = PaperPosition(
                symbol=order.symbol,
                quantity=order.quantity,
                side='long',
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price,
                highest_price=price,
                commission_paid=commission
            )
            self.account.positions[order.symbol] = position
    
    def _execute_sell(self, order: PaperOrder, price: float, commission: float, slippage: float):
        """Execute sell order"""
        if order.symbol in self.account.positions:
            pos = self.account.positions[order.symbol]
            
            if pos.side == 'long':
                # Closing long position
                if order.quantity >= pos.quantity:
                    # Close entire position
                    proceeds = pos.quantity * price - commission
                    pnl = proceeds - (pos.quantity * pos.entry_price + pos.commission_paid)
                    
                    # Record trade
                    self._record_trade(pos, price, pnl)
                    
                    # Update account
                    self.account.current_balance += proceeds
                    self.account.buying_power += proceeds
                    
                    # Remove position
                    del self.account.positions[order.symbol]
                else:
                    # Partial close
                    proceeds = order.quantity * price - commission
                    partial_pnl = (price - pos.entry_price) * order.quantity - commission
                    
                    # Update position
                    pos.quantity -= order.quantity
                    
                    # Update account
                    self.account.current_balance += proceeds
                    self.account.buying_power += proceeds
            else:
                # Adding to short position
                pos.quantity -= order.quantity
                pos.commission_paid += commission
        else:
            # New short position
            if self.allow_shorting:
                # Check margin requirement
                margin_required = order.quantity * price * self.margin_requirement
                if margin_required > self.account.available_margin:
                    order.status = OrderStatus.REJECTED
                    order.rejection_reason = "Insufficient margin"
                    return
                
                # Create short position
                position = PaperPosition(
                    symbol=order.symbol,
                    quantity=-order.quantity,
                    side='short',
                    entry_price=price,
                    entry_time=datetime.now(),
                    current_price=price,
                    lowest_price=price,
                    commission_paid=commission
                )
                self.account.positions[order.symbol] = position
                
                # Credit account (short sale proceeds)
                proceeds = order.quantity * price - commission
                self.account.current_balance += proceeds
    
    def _check_pending_orders(self, symbol: str):
        """Check pending orders for execution"""
        orders_to_execute = []
        
        for order_id, order in self.pending_orders.items():
            if order.symbol != symbol:
                continue
            
            # Check order type
            if order.order_type == OrderType.LIMIT:
                current_price = self.market_simulator.current_prices.get(symbol, 0)
                if order.side == OrderSide.BUY and current_price <= order.price:
                    orders_to_execute.append(order)
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    orders_to_execute.append(order)
            
            elif order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
                if self.market_simulator.check_stop_trigger(symbol, order.stop_price, order.side):
                    orders_to_execute.append(order)
        
        # Execute triggered orders
        for order in orders_to_execute:
            self._execute_order(order)
    
    def _update_positions(self, symbol: str):
        """Update position prices"""
        if symbol in self.account.positions:
            current_price = self.market_simulator.current_prices.get(symbol, 0)
            if current_price > 0:
                self.account.positions[symbol].update_price(current_price)
    
    def _check_exits(self, symbol: str):
        """Check stop loss and take profit exits"""
        if symbol not in self.account.positions:
            return
        
        pos = self.account.positions[symbol]
        current_price = pos.current_price
        
        # Check stop loss
        if pos.stop_loss:
            if (pos.side == 'long' and current_price <= pos.stop_loss) or \
               (pos.side == 'short' and current_price >= pos.stop_loss):
                self._exit_position(symbol, 'stop_loss')
                return
        
        # Check take profit
        if pos.take_profit:
            if (pos.side == 'long' and current_price >= pos.take_profit) or \
               (pos.side == 'short' and current_price <= pos.take_profit):
                self._exit_position(symbol, 'take_profit')
    
    def _exit_position(self, symbol: str, reason: str):
        """Exit a position"""
        pos = self.account.positions[symbol]
        
        # Create market order to close
        close_order = Order(
            symbol=symbol,
            side=OrderSide.SELL if pos.side == 'long' else OrderSide.BUY,
            quantity=abs(pos.quantity),
            order_type=OrderType.MARKET
        )
        
        # Place order
        order_id = self.place_order(close_order)
        logger.info(f"Position closed via {reason}: {symbol}")
    
    def _record_trade(self, position: PaperPosition, exit_price: float, pnl: float):
        """Record completed trade"""
        trade = {
            'symbol': position.symbol,
            'side': position.side,
            'quantity': position.quantity,
            'entry_price': position.entry_price,
            'entry_time': position.entry_time,
            'exit_price': exit_price,
            'exit_time': datetime.now(),
            'pnl': pnl,
            'pnl_pct': pnl / (position.quantity * position.entry_price),
            'commission': position.commission_paid,
            'holding_period': (datetime.now() - position.entry_time).total_seconds() / 3600
        }
        
        self.account.trades.append(trade)
        self.performance_analyzer.add_trade(trade)
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        # Calculate daily P&L
        today = datetime.now().date()
        daily_pnl = sum(
            trade['pnl'] for trade in self.account.trades 
            if trade.get('exit_time') and trade['exit_time'].date() == today
        )
        
        return {
            'balance': self.account.current_balance,
            'equity': self.account.total_equity,
            'buying_power': self.account.buying_power,
            'positions': len(self.account.positions),
            'open_orders': len(self.pending_orders),
            'total_trades': len(self.account.trades),
            'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.account.positions.values()),
            'realized_pnl': sum(trade['pnl'] for trade in self.account.trades),
            'daily_pnl': daily_pnl,
            'performance_metrics': self._get_performance_metrics_dict()
        }
    
    def _get_performance_metrics_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary"""
        try:
            metrics = self.performance_analyzer.calculate_metrics()
            return {
                'total_return': metrics.total_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'max_drawdown': metrics.max_drawdown,
                'profit_factor': metrics.profit_factor
            }
        except:
            # Return default metrics if no trades yet
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0
            }
    
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions"""
        positions = {}
        
        for symbol, pos in self.account.positions.items():
            positions[symbol] = {
                'quantity': pos.quantity,
                'side': pos.side,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'unrealized_pnl_pct': pos.unrealized_pnl / (abs(pos.quantity) * pos.entry_price),
                'stop_loss': pos.stop_loss,
                'take_profit': pos.take_profit
            }
        
        return positions
    
    def save_state(self):
        """Save paper trading state to file"""
        state = {
            'account': {
                'initial_balance': self.account.initial_balance,
                'current_balance': self.account.current_balance,
                'buying_power': self.account.buying_power
            },
            'positions': {
                symbol: {
                    'quantity': pos.quantity,
                    'side': pos.side,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time.isoformat(),
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'commission_paid': pos.commission_paid
                }
                for symbol, pos in self.account.positions.items()
            },
            'trades': self.account.trades,
            'order_counter': self.order_counter
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self):
        """Load paper trading state from file"""
        if not self.state_file.exists():
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Restore account
            self.account.initial_balance = state['account']['initial_balance']
            self.account.current_balance = state['account']['current_balance']
            self.account.buying_power = state['account']['buying_power']
            
            # Restore positions
            for symbol, pos_data in state['positions'].items():
                position = PaperPosition(
                    symbol=symbol,
                    quantity=pos_data['quantity'],
                    side=pos_data['side'],
                    entry_price=pos_data['entry_price'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']),
                    current_price=pos_data['entry_price'],  # Will be updated
                    stop_loss=pos_data.get('stop_loss'),
                    take_profit=pos_data.get('take_profit'),
                    commission_paid=pos_data.get('commission_paid', 0)
                )
                self.account.positions[symbol] = position
            
            # Restore trades
            self.account.trades = state['trades']
            for trade in self.account.trades:
                self.performance_analyzer.add_trade(trade)
            
            # Restore counter
            self.order_counter = state.get('order_counter', 0)
            
            logger.info("Paper trading state loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    def reset(self):
        """Reset paper trading account"""
        self.account = PaperAccount(
            initial_balance=self.account.initial_balance,
            current_balance=self.account.initial_balance,
            buying_power=self.account.initial_balance
        )
        self.pending_orders.clear()
        self.order_counter = 0
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Delete state file
        if self.state_file.exists():
            self.state_file.unlink()
        
        logger.info("Paper trading account reset")

class PaperTradingAPI:
    """API endpoints for paper trading"""
    
    def __init__(self, engine: PaperTradingEngine):
        self.engine = engine
    
    async def get_account(self) -> Dict[str, Any]:
        """Get account information"""
        return self.engine.get_account_summary()
    
    async def get_positions(self) -> Dict[str, Any]:
        """Get current positions"""
        return self.engine.get_positions()
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get pending orders"""
        orders = []
        for order in self.engine.pending_orders.values():
            orders.append({
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'order_type': order.order_type.value,
                'price': order.price,
                'status': order.status.value,
                'created_at': order.created_at.isoformat()
            })
        return orders
    
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a new order"""
        order = Order(
            symbol=order_data['symbol'],
            side=OrderSide[order_data['side']],
            quantity=order_data['quantity'],
            order_type=OrderType[order_data.get('order_type', 'MARKET')],
            price=order_data.get('price'),
            stop_price=order_data.get('stop_price')
        )
        
        order_id = self.engine.place_order(order)
        
        return {
            'success': bool(order_id),
            'order_id': order_id
        }
    
    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an order"""
        success = self.engine.cancel_order(order_id)
        return {'success': success}
    
    async def get_performance(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.engine.performance_analyzer.calculate_metrics()
        return metrics.to_dict()
    
    async def reset_account(self) -> Dict[str, Any]:
        """Reset paper trading account"""
        self.engine.reset()
        return {'success': True, 'message': 'Account reset successfully'}