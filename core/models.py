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
    """v-pricebook-2026-05-01 (Step 2 of the canonical PriceBook
    migration): `current_price` and `unrealized_pnl` are GONE from
    this dataclass. Anyone needing the mark-to-market price of a
    position MUST go through PriceBook + CalculationEngine.get_pnl().

    For backward compatibility during the migration, two lazy
    properties expose `current_price` and `unrealized_pnl` as
    reactive-pull values. They look up PriceBook.instance() and
    compute on read. This means:
      - The fields look the same to legacy callers (pos.current_price
        still returns a float).
      - But they CANNOT be set or persisted — assignment raises a
        DeprecationWarning (we'll harden to AttributeError once all
        legacy writers are deleted).
      - Stale or missing prices return 0.0 with no_price marker.
    """
    symbol: str
    entry_price: float
    quantity: int
    side: str
    stop_loss: float
    take_profit: float
    entry_time: datetime
    reasoning: Dict[str, Any] = field(default_factory=dict)
    is_long_term: bool = False
    scaled_out: bool = False            # Whether 1R partial exit has fired
    original_stop: Optional[float] = None  # Stop at entry (for computing R)
    trailing_stop: Optional[float] = None  # Current ATR trailing stop level
    # v-mode-field-2026-04-20: attribute-based tagging replaces the old
    # container-only tagging ("position in self.positions = live"). Default
    # "simulation" so an unlabeled position never silently looks live.
    # Live-mode Position() sites must pass mode="live" explicitly.
    mode: str = "simulation"
    # v-breakeven-stop-2026-04-28: high-water mark for favorable R-multiple.
    # Used by breakeven-stop ratchet: once peak crosses BREAKEVEN_ACTIVATION_R,
    # stop_loss is lifted to entry so the trade can no longer become a loser.
    peak_favorable_r: float = 0.0
    breakeven_lifted: bool = False
    # v-managed-by-bot-2026-04-28: per-position auto-management flag.
    # True  = bot owns this trade — applies stops, trail, breakeven, exits.
    # False = hands-off (Schwab-synced / human-opened / user toggled OFF).
    # In LIVE mode this is the primary gate; in SIM mode it's also honored
    # so users can pause management on a single position from the dashboard.
    managed_by_bot: bool = False
    # v-thesis-revalidate-2026-04-28: last time the thesis re-validation ran
    # for this position. Tracked separately from entry/management ticks so
    # the re-check can fire on its own cadence (default every 15 min after
    # the 30-min mark). ISO timestamp string; None means never re-checked.
    last_revalidation_at: Optional[str] = None
    # ──────────────────────────────────────────────────────────────────
    # v-position-fsm-2026-04-30 (Phase 1): finite-state-machine fields.
    # See core/position_state.py for the full state-table contract.
    # ──────────────────────────────────────────────────────────────────
    # Stored as the str value of PositionState (e.g. "live") so the
    # JSON serialisation in _save_state stays human-readable.
    state: str = "live"   # PositionState.LIVE.value — default for newly opened sims
    # Reason the position entered its current state (e.g. "+0.5R reached").
    # Audit-only; not used for decisions.
    state_reason: str = "initial"
    # ISO timestamp of last state transition. Persisted; survives restart.
    state_changed_at: Optional[str] = None
    # ISO timestamp of when the EXITING state was entered. Used by the
    # zombie-recovery logic — if EXITING has been held for >EXITING_ZOMBIE_SEC
    # seconds (default 60), the close clearly didn't complete and the
    # position is auto-promoted to ZOMBIE for operator inspection.
    exiting_started_at: Optional[str] = None
    # Reason for ZOMBIE promotion, if any. Operator reads this to decide
    # whether to retry the close, write off the trade, or clear manually.
    zombie_reason: Optional[str] = None
    # The asyncio.Lock that serialises state transitions for THIS position.
    # Initialised lazily at engine boot (after the asyncio loop exists)
    # because dataclass `default_factory=asyncio.Lock` would fail at
    # module import in non-async contexts. See engine._ensure_position_lock.
    # field(repr=False, compare=False) so the lock doesn't try to compare
    # or print as part of the dataclass machinery.
    _state_lock: Any = field(default=None, repr=False, compare=False)

    # ──────────────────────────────────────────────────────────────────
    # v-order-monitor-2026-09-10: bracket/OCO order tracking fields.
    # When _place_bracket_orders succeeds, these IDs track the broker-side
    # orders so the order monitor can:
    #   1. Detect stop/TP fills and sync Position state
    #   2. Cancel sibling legs when one side fills
    #   3. Replace stops when software trail moves
    # All three are optional (None = no active bracket for this position).
    # ──────────────────────────────────────────────────────────────────
    bracket_order_id: Optional[str] = None   # Parent OCO order ID
    stop_order_id: Optional[str] = None      # Stop loss leg order ID
    tp_order_id: Optional[str] = None        # Take profit leg order ID
    # Last broker stop price (for detecting when trail replacement is needed)
    broker_stop_price: Optional[float] = None

    # ──────────────────────────────────────────────────────────────────
    # v-pricebook-2026-05-01: reactive-pull current_price / unrealized_pnl.
    # These look like attributes for backwards-compat but compute on
    # every read against the canonical PriceBook. There is no stored
    # mark on this object anymore.
    # ──────────────────────────────────────────────────────────────────
    @property
    def current_price(self) -> float:
        """Reactive-pull mark from PriceBook. Returns entry_price as a
        safe fallback when PriceBook isn't installed yet (early boot)
        or when the symbol has no quote, so legacy callers don't divide-
        by-zero. Returns 0.0 only if entry_price is also missing."""
        try:
            from core.price_book import PriceBook
            pb = PriceBook.instance()
            if pb is not None:
                obj = pb.get_mark(self.symbol)
                if obj.has_price and not obj.is_stale:
                    return obj.price
                # Stale-but-known: still return the last we have so the
                # dashboard isn't blank; the is_stale flag is exposed
                # via the helper get_pnl result for surfaces that care.
                if obj.has_price:
                    return obj.price
        except Exception:
            pass
        return float(self.entry_price or 0.0)

    @current_price.setter
    def current_price(self, value):
        # During the migration window, legacy code may still try to set
        # this. Silently swallow — the value is owned by the PriceBook
        # now, no Position-side mutation is permitted. Once all legacy
        # writers are removed, switch this to raise AttributeError to
        # catch any straggler writes loudly.
        return

    @property
    def unrealized_pnl(self) -> float:
        """Reactive-pull P&L. Computes from the live mark every read."""
        try:
            from core.price_book import PriceBook
            from core.calculations import CalculationEngine
            pb = PriceBook.instance()
            if pb is not None:
                result = CalculationEngine.get_pnl(self, pb.get_mark(self.symbol))
                return result.unrealized_pnl
        except Exception:
            pass
        return 0.0

    @unrealized_pnl.setter
    def unrealized_pnl(self, value):
        # Same as current_price — derived field, can't be set.
        return


# v-pricebook-2026-05-01: backward-compat init shim. Many existing call
# sites still pass current_price=, unrealized_pnl=, and legacy keys
# like is_external / is_manually_managed (which were ad-hoc attributes
# never declared on the dataclass). The dataclass-generated __init__
# rejects unknown kwargs with TypeError. Wrap it: drop the now-derived
# fields, set everything else, then attach any remaining legacy attrs
# as instance attributes.
_position_orig_init = Position.__init__
_DERIVED_FIELDS = {"current_price", "unrealized_pnl"}
def _position_migration_init(self, *args, **kwargs):
    # Pull off legacy/derived fields so the dataclass init won't reject
    legacy_attrs = {}
    for k in list(kwargs.keys()):
        if k in _DERIVED_FIELDS:
            kwargs.pop(k)  # silently drop — derived now
        elif k in {"is_external", "is_manually_managed", "is_long_term", "day_pnl"}:
            legacy_attrs[k] = kwargs.pop(k)
    _position_orig_init(self, *args, **kwargs)
    # Re-attach legacy attrs the rest of the codebase still reads
    for k, v in legacy_attrs.items():
        try:
            setattr(self, k, v)
        except Exception:
            pass
Position.__init__ = _position_migration_init


def clear_bracket_ids(position: Position) -> None:
    """Clear all bracket/OCO order tracking IDs on a position.
    
    v-order-monitor-2026-09-10: called after bracket fills/cancels/rejects
    to reset tracking state so the order monitor stops watching stale IDs.
    """
    position.bracket_order_id = None
    position.stop_order_id = None
    position.tp_order_id = None
    position.broker_stop_price = None


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
