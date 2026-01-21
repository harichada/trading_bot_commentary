"""
Trading State Management

Immutable state objects for positions, orders, and portfolio state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side"""
    LONG = "long"
    SHORT = "short"


@dataclass
class Order:
    """Immutable order representation"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    broker_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if order is still active"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def is_complete(self) -> bool:
        """Check if order is complete"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'broker_order_id': self.broker_order_id,
            'metadata': self.metadata,
        }


@dataclass
class Position:
    """Immutable position representation"""
    symbol: str
    side: PositionSide
    quantity: int
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    strategy: str = "unknown"
    is_external: bool = False  # True if position was opened outside the bot
    is_manually_managed: bool = False  # True if bot shouldn't auto-manage
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L"""
        if self.side == PositionSide.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.entry_price == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.current_price * self.quantity

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis"""
        return self.entry_price * self.quantity

    def with_price(self, new_price: float) -> 'Position':
        """Return new position with updated price"""
        return Position(
            symbol=self.symbol,
            side=self.side,
            quantity=self.quantity,
            entry_price=self.entry_price,
            current_price=new_price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            entry_time=self.entry_time,
            strategy=self.strategy,
            is_external=self.is_external,
            is_manually_managed=self.is_manually_managed,
            metadata=self.metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'strategy': self.strategy,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'market_value': self.market_value,
            'is_external': self.is_external,
            'is_manually_managed': self.is_manually_managed,
        }


@dataclass
class PortfolioState:
    """Current portfolio state snapshot"""
    timestamp: datetime
    account_balance: float
    buying_power: float
    cash: float
    positions: Dict[str, Position]
    pending_orders: Dict[str, Order]
    daily_pnl: float = 0.0
    total_pnl: float = 0.0

    @property
    def total_market_value(self) -> float:
        """Total market value of all positions"""
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def position_count(self) -> int:
        """Number of open positions"""
        return len(self.positions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'account_balance': self.account_balance,
            'buying_power': self.buying_power,
            'cash': self.cash,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'pending_orders': {k: v.to_dict() for k, v in self.pending_orders.items()},
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl,
            'total_market_value': self.total_market_value,
            'total_unrealized_pnl': self.total_unrealized_pnl,
            'position_count': self.position_count,
        }


@dataclass
class TradingState:
    """Complete trading state"""
    mode: str  # simulation, paper, live
    is_running: bool
    portfolio: PortfolioState
    trade_history: List[Dict[str, Any]] = field(default_factory=list)
    last_analysis_time: Optional[datetime] = None
    consecutive_losses: int = 0
    circuit_breaker_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'mode': self.mode,
            'is_running': self.is_running,
            'portfolio': self.portfolio.to_dict(),
            'trade_count': len(self.trade_history),
            'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self.circuit_breaker_active,
        }

    def save(self, path: str = "data/trading_state.json"):
        """Save state to file"""
        from pathlib import Path
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = "data/trading_state.json") -> Optional['TradingState']:
        """Load state from file"""
        from pathlib import Path
        filepath = Path(path)

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # Would need to reconstruct objects from dict
            # For now, return None to start fresh
            return None
        except Exception:
            return None
