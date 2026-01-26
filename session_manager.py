"""
Session Manager - Market Session Awareness System
Identifies: Pre-market, Open, Prime Time, Dead Zone, Power Hour, After Hours
Adjusts trading parameters based on time of day

A pro knows WHEN to trade is as important as WHAT to trade.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, time, timedelta
import pytz
import logging

logger = logging.getLogger(__name__)


class MarketSession(Enum):
    """Market session classifications"""
    PRE_MARKET = "pre_market"          # 4:00 AM - 9:30 AM ET
    MARKET_OPEN = "market_open"        # 9:30 AM - 10:00 AM ET (volatile, traps)
    MORNING_MOMENTUM = "morning_momentum"  # 10:00 AM - 11:30 AM ET (best trends)
    LUNCH_DEAD_ZONE = "lunch_dead_zone"    # 11:30 AM - 2:00 PM ET (chop, avoid)
    AFTERNOON = "afternoon"            # 2:00 PM - 3:00 PM ET (positioning)
    POWER_HOUR = "power_hour"          # 3:00 PM - 4:00 PM ET (institutional moves)
    AFTER_HOURS = "after_hours"        # 4:00 PM - 8:00 PM ET
    CLOSED = "closed"                  # Market closed


@dataclass
class SessionState:
    """Current session state with trading parameters"""
    session: MarketSession
    session_name: str
    minutes_into_session: int
    minutes_until_next_session: int
    allow_new_trades: bool
    allow_exits: bool
    position_size_multiplier: float  # 0.0 to 1.5
    volatility_expected: str  # low, normal, high
    spread_warning: bool  # Wider spreads expected
    institutional_activity: str  # low, normal, high
    recommendation: str
    timestamp: datetime


@dataclass
class SessionParameters:
    """Trading parameters for each session"""
    allow_entries: bool
    allow_exits: bool
    position_size_mult: float
    min_confidence: float
    expected_volatility: str
    spread_expectation: str
    institutional_activity: str
    max_trades: int
    stop_buffer_mult: float  # Wider stops during volatile sessions
    notes: str


class SessionManager:
    """
    Manages trading based on market session timing.

    Philosophy:
    - Market Open (9:30-10:00): AVOID - fakeouts, gap fills, retail traps
    - Morning Momentum (10:00-11:30): PRIME TIME - best trends develop
    - Lunch (11:30-2:00): DEAD ZONE - low volume, chop, reversals
    - Afternoon (2:00-3:00): CAUTION - positioning for close
    - Power Hour (3:00-4:00): GOOD - institutional activity, real moves

    The best traders know that making money is about WHEN you trade,
    not just what you trade.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.timezone = pytz.timezone(self.config.get('timezone', 'US/Eastern'))

        # Customizable session times (in ET)
        self.session_times = {
            MarketSession.PRE_MARKET: (time(4, 0), time(9, 30)),
            MarketSession.MARKET_OPEN: (time(9, 30), time(10, 0)),
            MarketSession.MORNING_MOMENTUM: (time(10, 0), time(11, 30)),
            MarketSession.LUNCH_DEAD_ZONE: (time(11, 30), time(14, 0)),
            MarketSession.AFTERNOON: (time(14, 0), time(15, 0)),
            MarketSession.POWER_HOUR: (time(15, 0), time(16, 0)),
            MarketSession.AFTER_HOURS: (time(16, 0), time(20, 0)),
        }

        # Session-specific parameters
        self._setup_session_parameters()

        # State tracking
        self.current_session: Optional[SessionState] = None
        self.trades_this_session: int = 0
        self.session_start_time: Optional[datetime] = None

        # Special days (earnings, FOMC, etc.)
        self.special_days: Dict[str, str] = {}  # date -> event type

    def _setup_session_parameters(self):
        """Define trading parameters for each session"""
        self.session_params = {
            MarketSession.PRE_MARKET: SessionParameters(
                allow_entries=False,  # No new positions pre-market
                allow_exits=True,     # Can exit if needed
                position_size_mult=0.0,
                min_confidence=0.95,  # Effectively block
                expected_volatility="low",
                spread_expectation="wide",
                institutional_activity="low",
                max_trades=0,
                stop_buffer_mult=2.0,  # If somehow trading, wide stops
                notes="Pre-market: Low liquidity, wide spreads. AVOID NEW POSITIONS."
            ),

            MarketSession.MARKET_OPEN: SessionParameters(
                allow_entries=False,  # First 30 min = trap city
                allow_exits=True,
                position_size_mult=0.25,  # Minimal if overridden
                min_confidence=0.90,
                expected_volatility="high",
                spread_expectation="normal",
                institutional_activity="high",
                max_trades=0,  # Don't trade the open
                stop_buffer_mult=1.5,
                notes="Market Open: High volatility, gap fills, retail traps. WAIT FOR CLARITY."
            ),

            MarketSession.MORNING_MOMENTUM: SessionParameters(
                allow_entries=True,  # PRIME TIME
                allow_exits=True,
                position_size_mult=1.0,  # Full size allowed
                min_confidence=0.65,
                expected_volatility="normal",
                spread_expectation="tight",
                institutional_activity="high",
                max_trades=4,
                stop_buffer_mult=1.0,
                notes="Morning Momentum: BEST TIME TO TRADE. Trends establish, follow them."
            ),

            MarketSession.LUNCH_DEAD_ZONE: SessionParameters(
                allow_entries=False,  # AVOID
                allow_exits=True,
                position_size_mult=0.0,
                min_confidence=0.95,
                expected_volatility="low",
                spread_expectation="normal",
                institutional_activity="low",
                max_trades=0,
                stop_buffer_mult=1.0,
                notes="Lunch Dead Zone: Low volume, choppy, reversals. STAND ASIDE."
            ),

            MarketSession.AFTERNOON: SessionParameters(
                allow_entries=True,  # Cautious entries
                allow_exits=True,
                position_size_mult=0.5,
                min_confidence=0.75,
                expected_volatility="normal",
                spread_expectation="normal",
                institutional_activity="normal",
                max_trades=2,
                stop_buffer_mult=1.0,
                notes="Afternoon: Institutional positioning begins. Be selective."
            ),

            MarketSession.POWER_HOUR: SessionParameters(
                allow_entries=True,  # Good trading
                allow_exits=True,
                position_size_mult=0.8,
                min_confidence=0.70,
                expected_volatility="high",
                spread_expectation="normal",
                institutional_activity="high",
                max_trades=2,
                stop_buffer_mult=1.2,
                notes="Power Hour: Institutional activity, real moves. Trade with the flow."
            ),

            MarketSession.AFTER_HOURS: SessionParameters(
                allow_entries=False,
                allow_exits=True,  # Exit if needed
                position_size_mult=0.0,
                min_confidence=0.95,
                expected_volatility="low",
                spread_expectation="wide",
                institutional_activity="low",
                max_trades=0,
                stop_buffer_mult=2.0,
                notes="After Hours: Low liquidity, wide spreads. AVOID."
            ),

            MarketSession.CLOSED: SessionParameters(
                allow_entries=False,
                allow_exits=False,
                position_size_mult=0.0,
                min_confidence=1.0,
                expected_volatility="none",
                spread_expectation="none",
                institutional_activity="none",
                max_trades=0,
                stop_buffer_mult=1.0,
                notes="Market Closed."
            ),
        }

    def get_current_time_et(self) -> datetime:
        """Get current time in Eastern timezone"""
        return datetime.now(self.timezone)

    def is_market_open(self) -> bool:
        """Check if regular market hours (9:30 AM - 4:00 PM ET)"""
        now = self.get_current_time_et()

        # Check weekday (0 = Monday, 6 = Sunday)
        if now.weekday() >= 5:
            return False

        current_time = now.time()
        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= current_time < market_close

    def get_current_session(self) -> MarketSession:
        """Determine current market session"""
        now = self.get_current_time_et()

        # Weekend
        if now.weekday() >= 5:
            return MarketSession.CLOSED

        current_time = now.time()

        # Check each session
        for session, (start, end) in self.session_times.items():
            if start <= current_time < end:
                return session

        # Before pre-market or after after-hours
        return MarketSession.CLOSED

    def get_session_state(self) -> SessionState:
        """Get comprehensive session state"""
        now = self.get_current_time_et()
        session = self.get_current_session()
        params = self.session_params[session]

        # Calculate minutes into current session
        minutes_into = 0
        minutes_until_next = 0

        if session in self.session_times:
            start_time, end_time = self.session_times[session]
            session_start = now.replace(
                hour=start_time.hour,
                minute=start_time.minute,
                second=0,
                microsecond=0
            )
            minutes_into = int((now - session_start).total_seconds() / 60)
            session_end = now.replace(
                hour=end_time.hour,
                minute=end_time.minute,
                second=0,
                microsecond=0
            )
            minutes_until_next = int((session_end - now).total_seconds() / 60)

        # Track session changes
        if self.current_session is None or self.current_session.session != session:
            self.trades_this_session = 0
            self.session_start_time = now
            logger.info(f"Session changed to: {session.value}")

        state = SessionState(
            session=session,
            session_name=self._get_session_name(session),
            minutes_into_session=minutes_into,
            minutes_until_next_session=minutes_until_next,
            allow_new_trades=params.allow_entries and self.trades_this_session < params.max_trades,
            allow_exits=params.allow_exits,
            position_size_multiplier=params.position_size_mult,
            volatility_expected=params.expected_volatility,
            spread_warning=params.spread_expectation == "wide",
            institutional_activity=params.institutional_activity,
            recommendation=params.notes,
            timestamp=now
        )

        self.current_session = state
        return state

    def _get_session_name(self, session: MarketSession) -> str:
        """Get human-readable session name"""
        names = {
            MarketSession.PRE_MARKET: "Pre-Market",
            MarketSession.MARKET_OPEN: "Market Open (AVOID)",
            MarketSession.MORNING_MOMENTUM: "Morning Momentum (PRIME TIME)",
            MarketSession.LUNCH_DEAD_ZONE: "Lunch Dead Zone (AVOID)",
            MarketSession.AFTERNOON: "Afternoon Session",
            MarketSession.POWER_HOUR: "Power Hour",
            MarketSession.AFTER_HOURS: "After Hours",
            MarketSession.CLOSED: "Market Closed",
        }
        return names.get(session, "Unknown")

    def should_trade(self, trade_type: str = "entry") -> Tuple[bool, str]:
        """
        Check if trading is allowed in current session.

        Args:
            trade_type: "entry" for new positions, "exit" for closing

        Returns:
            (allowed, reason)
        """
        state = self.get_session_state()
        params = self.session_params[state.session]

        if trade_type == "entry":
            if not params.allow_entries:
                return False, f"New entries blocked during {state.session_name}"

            if self.trades_this_session >= params.max_trades:
                return False, f"Max trades ({params.max_trades}) reached for {state.session_name}"

            if state.spread_warning:
                return False, f"Wide spreads expected during {state.session_name}"

            return True, f"Entries allowed: {state.session_name}"

        else:  # exit
            if not params.allow_exits:
                return False, f"Market closed - exits not possible"

            return True, f"Exits allowed: {state.session_name}"

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier for current session"""
        state = self.get_session_state()
        return state.position_size_multiplier

    def get_stop_buffer_multiplier(self) -> float:
        """Get stop loss buffer multiplier for current session volatility"""
        state = self.get_session_state()
        return self.session_params[state.session].stop_buffer_mult

    def get_min_confidence(self) -> float:
        """Get minimum confidence threshold for current session"""
        state = self.get_session_state()
        return self.session_params[state.session].min_confidence

    def record_trade(self):
        """Record that a trade was made this session"""
        self.trades_this_session += 1
        logger.info(f"Trade recorded. Trades this session: {self.trades_this_session}")

    def is_prime_time(self) -> bool:
        """Check if we're in prime trading time (Morning Momentum)"""
        session = self.get_current_session()
        return session == MarketSession.MORNING_MOMENTUM

    def is_dead_zone(self) -> bool:
        """Check if we're in a dead zone (lunch, pre-market, after-hours)"""
        session = self.get_current_session()
        return session in [
            MarketSession.LUNCH_DEAD_ZONE,
            MarketSession.PRE_MARKET,
            MarketSession.AFTER_HOURS,
            MarketSession.MARKET_OPEN,  # First 30 min is also dangerous
            MarketSession.CLOSED
        ]

    def time_until_prime_time(self) -> Optional[timedelta]:
        """Get time until next prime trading session"""
        now = self.get_current_time_et()

        # If weekend, calculate to Monday 10:00 AM
        if now.weekday() >= 5:
            days_until_monday = 7 - now.weekday()
            next_prime = now.replace(
                hour=10, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_until_monday)
            return next_prime - now

        current_time = now.time()
        prime_start = time(10, 0)
        prime_end = time(11, 30)

        if current_time < prime_start:
            # Before prime time today
            prime_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
            return prime_datetime - now
        elif prime_start <= current_time < prime_end:
            # In prime time now
            return timedelta(0)
        else:
            # After prime time, next is tomorrow (or Monday)
            if now.weekday() == 4:  # Friday
                days_ahead = 3
            else:
                days_ahead = 1
            next_prime = now.replace(
                hour=10, minute=0, second=0, microsecond=0
            ) + timedelta(days=days_ahead)
            return next_prime - now

    def add_special_day(self, date: str, event: str):
        """
        Mark a day as special (FOMC, earnings, etc.)

        Args:
            date: "YYYY-MM-DD" format
            event: Description of event
        """
        self.special_days[date] = event
        logger.info(f"Special day added: {date} - {event}")

    def is_special_day(self) -> Tuple[bool, Optional[str]]:
        """Check if today is a special trading day"""
        today = self.get_current_time_et().strftime("%Y-%m-%d")
        if today in self.special_days:
            return True, self.special_days[today]
        return False, None

    def get_session_summary(self) -> Dict:
        """Get human-readable session summary for commentary"""
        state = self.get_session_state()
        params = self.session_params[state.session]

        is_special, special_event = self.is_special_day()

        return {
            "session": state.session.value,
            "session_name": state.session_name,
            "time_et": state.timestamp.strftime("%H:%M ET"),
            "minutes_into_session": state.minutes_into_session,
            "minutes_remaining": state.minutes_until_next_session,
            "allow_entries": state.allow_new_trades,
            "allow_exits": state.allow_exits,
            "trades_this_session": self.trades_this_session,
            "max_trades_allowed": params.max_trades,
            "position_size_mult": state.position_size_multiplier,
            "volatility_expected": state.volatility_expected,
            "institutional_activity": state.institutional_activity,
            "is_prime_time": self.is_prime_time(),
            "is_dead_zone": self.is_dead_zone(),
            "is_special_day": is_special,
            "special_event": special_event,
            "recommendation": state.recommendation,
            "message": self._get_session_message(state)
        }

    def _get_session_message(self, state: SessionState) -> str:
        """Generate session-specific trading guidance"""
        messages = {
            MarketSession.PRE_MARKET:
                "Pre-market session. Low liquidity, wide spreads. Review overnight gaps and news. NO NEW POSITIONS.",

            MarketSession.MARKET_OPEN:
                "First 30 minutes. High volatility, gap fills, retail traps everywhere. WAIT FOR THE DUST TO SETTLE.",

            MarketSession.MORNING_MOMENTUM:
                "PRIME TIME. Best trading window of the day. Trends establish here - trade with conviction.",

            MarketSession.LUNCH_DEAD_ZONE:
                "Lunch dead zone. Low volume, choppy action, mean reversion traps. Pros go to lunch. YOU SHOULD TOO.",

            MarketSession.AFTERNOON:
                "Afternoon positioning. Institutional players setting up for close. Be selective, follow the big money.",

            MarketSession.POWER_HOUR:
                "Power Hour. Institutional activity peaks. Real moves happen here. Trade with the flow, not against it.",

            MarketSession.AFTER_HOURS:
                "After hours. Thin liquidity, wide spreads, news-driven spikes. Exit if needed, no new positions.",

            MarketSession.CLOSED:
                "Market closed. Review your trades, plan for tomorrow, rest your mind."
        }
        return messages.get(state.session, "Unknown session")


# Singleton instance
_session_manager: Optional[SessionManager] = None

def get_session_manager(config: Optional[Dict] = None) -> SessionManager:
    """Get or create singleton SessionManager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(config)
    return _session_manager
