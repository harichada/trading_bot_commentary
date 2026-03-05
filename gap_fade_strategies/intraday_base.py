"""Abstract base class for intraday strategies.

Intraday strategies actively SCAN for setups throughout the trading day,
unlike GapFadeStrategy which filters pre-scanned gap candidates.

They share the same position pool, indicator engine, order execution,
and risk management as gap fade strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class IntradaySetup:
    """Signal from an intraday strategy to enter a position."""
    symbol: str
    strategy_id: str
    direction: str           # 'long' or 'short'
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward: float
    confidence: float        # 0.0 - 1.0
    setup_type: str          # e.g. 'orb_breakout', 'momentum_surge'
    indicators: Dict = field(default_factory=dict)  # indicator snapshot at signal time
    timestamp: Optional[datetime] = None
    notes: str = ''


class IntradayStrategy(ABC):
    """Base class for intraday strategies that scan for setups during the day.

    Convention: returning None means "no setup found" or "use default behavior".
    Subclasses implement scan_for_setups() to actively search for trade opportunities.
    """

    # Subclasses must set these
    name: str = ''
    description: str = ''
    version: str = '1.0'
    strategy_id: str = ''

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional strategy-specific config dict."""
        self._config = config or self.get_default_config()

    @property
    def config(self) -> Dict:
        return self._config

    # ── Scanning ──────────────────────────────────────────────────────

    @abstractmethod
    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        """Scan a symbol for trade setups.

        Called periodically for each symbol on the watchlist.

        Args:
            symbol: ticker symbol
            tick_data: indicator data from TickIndicatorEngine.get_data()
            snapshot: latest price/volume snapshot (from REST or stream)
            now: current datetime (ET)

        Returns:
            IntradaySetup if a setup is detected, None otherwise.
        """
        ...

    @abstractmethod
    def get_watchlist_criteria(self) -> Dict:
        """Return criteria for building the watchlist.

        Returns dict with filtering parameters:
            {
                'min_volume': int,        # minimum volume
                'min_price': float,       # minimum price
                'max_price': float,       # maximum price
                'min_adr_pct': float,     # minimum average daily range %
                'prefer_gappers': bool,   # prefer morning gap candidates
            }
        """
        ...

    # ── Entry ─────────────────────────────────────────────────────────

    def get_active_window(self) -> Tuple[int, int, int, int]:
        """Return (start_hour, start_min, end_hour, end_min) for active scanning.

        Default: 9:45 AM to 3:30 PM ET.
        """
        return (9, 45, 15, 30)

    def validate_setup(self, setup: IntradaySetup, tick_data: Dict,
                       now: datetime) -> Tuple[bool, str]:
        """Validate a setup before entry.

        Checks R:R ratio, confidence threshold, and timing.

        Args:
            setup: the proposed trade setup
            tick_data: current indicator data
            now: current datetime

        Returns:
            (valid, reason) — True to enter, False to skip with reason.
        """
        # Minimum R:R check
        min_rr = self._config.get('min_risk_reward', 1.5)
        if setup.risk_reward < min_rr:
            return False, f'R:R {setup.risk_reward:.1f} < minimum {min_rr}'

        # Minimum confidence check
        min_conf = self._config.get('min_confidence', 0.5)
        if setup.confidence < min_conf:
            return False, f'Confidence {setup.confidence:.2f} < minimum {min_conf}'

        # Active window check
        h1, m1, h2, m2 = self.get_active_window()
        if now.hour < h1 or (now.hour == h1 and now.minute < m1):
            return False, f'Before active window ({h1}:{m1:02d})'
        if now.hour > h2 or (now.hour == h2 and now.minute > m2):
            return False, f'After active window ({h2}:{m2:02d})'

        return True, ''

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional['ExitSignal']:
        """Check for strategy-specific exit conditions.

        Called periodically for open positions opened by this strategy.
        Returns ExitSignal to close/modify, None for no action.
        """
        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Compute a new trailing stop price.

        None = no change. Returns float = new stop price.
        """
        return None

    # ── Indicators ────────────────────────────────────────────────────

    @abstractmethod
    def get_required_indicators(self) -> List[str]:
        """List of indicators needed from TickIndicatorEngine.

        Valid values: 'vwap', 'ema', 'ema_multi', 'opening_range',
                      'day_high', 'day_low', 'bar_history',
                      'rsi', 'atr', 'volume_profile'
        """
        ...

    def get_ema_periods(self) -> List[int]:
        """Return EMA periods needed by this strategy.

        Default: [9, 20]. Override for different periods.
        """
        return [9, 20]

    # ── UI / Config ───────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        """Return JSON schema for strategy-specific parameters.

        Used by UI to render config fields. Format:
        {
            "param_name": {
                "type": "float" | "int" | "bool" | "str",
                "label": "Human Label",
                "default": value,
                "min": optional,
                "max": optional,
                "description": "tooltip text"
            }
        }
        """
        return {}

    def get_default_config(self) -> Dict:
        """Return default values for strategy-specific config."""
        return {}

    def update_config(self, new_config: Dict):
        """Update strategy-specific config (e.g., from UI)."""
        self._config.update(new_config)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def on_day_start(self):
        """Called at start of each trading day. Reset daily state here."""
        pass

    def on_day_end(self):
        """Called at end of each trading day. Cleanup here."""
        pass
