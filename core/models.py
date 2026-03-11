from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    SIMULATION_WITH_COMMENTARY = "simulation_commentary"


class CommentaryType(Enum):
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    DECISION = "decision"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    PSYCHOLOGY = "psychology"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    ANOMALY = "anomaly"
    INFO = "info"
    ACCOUNT_UPDATE = "account_update"
    ERROR = "error"

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class NewsImpact(Enum):
    BREAKING = "breaking"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradingSignal:
    symbol: str
    signal_type: SignalType
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: int
    reasoning: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class Position:
    symbol: str
    entry_price: float
    current_price: float
    quantity: int
    side: str
    stop_loss: float
    take_profit: float
    entry_time: datetime
    unrealized_pnl: float = 0
    reasoning: Dict[str, Any] = field(default_factory=dict)
    is_long_term: bool = False  # Flag for long-term holdings

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str
    indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class NewsItem:
    id: str
    symbol: str
    headline: str
    summary: str
    source: str
    url: str
    published_time: datetime
    sentiment_score: float = 0.0
    sentiment_confidence: float = 0.0
    impact: NewsImpact = NewsImpact.LOW
    relevance_score: float = 0.0

    def age_hours(self) -> float:
        return (datetime.now() - self.published_time).total_seconds() / 3600
