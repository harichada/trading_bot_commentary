"""Abstract base class for gap fade strategies.

Every method has a sensible default (return None = "use engine default").
Strategy #1 (classic) overrides nothing. Strategy #2 (VWAP) overrides everything.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class ExitSignal:
    """Signal from strategy to close/modify a position."""
    action: str          # 'close', 'tighten_stop'
    reason: str          # human-readable reason
    shares: float = 0    # shares to close (0 = all remaining)
    new_stop: float = 0  # for tighten_stop


class GapFadeStrategy(ABC):
    """Base class for all gap fade strategies.

    Convention: returning None means "use engine default behavior".
    This lets simple strategies (classic) be pure passthroughs with zero
    behavior change, while complex strategies (VWAP) override everything.
    """

    # Subclasses must set these
    name: str = ''
    description: str = ''
    version: str = '1.0'

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with optional strategy-specific config dict."""
        self._config = config or self.get_default_config()

    @property
    def config(self) -> Dict:
        return self._config

    # ── Scanning ──────────────────────────────────────────────────────

    def filter_candidate(self, candidate: dict) -> Tuple[bool, str]:
        """Pre-filter a candidate before ranking.

        Args:
            candidate: dict from GapCandidate (asdict)

        Returns:
            (pass, reason) — True to keep, False to skip with reason.
            Default: always passes.
        """
        return True, ''

    def score_candidate(self, candidate: dict, regime: Optional[dict] = None) -> Optional[float]:
        """Adjust candidate score. Return None to keep engine score, or a float to override."""
        return None

    # ── Entry ─────────────────────────────────────────────────────────

    def get_entry_window(self) -> Optional[Tuple[int, int, int, int]]:
        """Return (start_hour, start_min, end_hour, end_min) for entry window.

        None = use engine defaults (9:31 to entry_cutoff).
        """
        return None

    def should_enter_now(self, candidate: dict, price: float,
                         tick_data: Optional[dict] = None,
                         now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Gate on whether to enter right now.

        Args:
            candidate: GapCandidate as dict
            price: current price
            tick_data: real-time indicator data from TickIndicatorEngine (VWAP, EMA, OR, etc.)
            now: current datetime (ET)

        Returns:
            (enter, reason) — True to enter, False to skip with reason.
            Default: always enters.
        """
        return True, ''

    # ── Stop & Targets ────────────────────────────────────────────────

    def compute_stop_price(self, entry_price: float, candidate: dict,
                           tick_data: Optional[dict] = None) -> Optional[float]:
        """Compute stop price. None = use engine default (fixed % or adaptive)."""
        return None

    def compute_targets(self, entry_price: float, candidate: dict,
                        tick_data: Optional[dict] = None) -> Optional[Tuple[float, float]]:
        """Compute (half_target, full_target). None = use engine defaults."""
        return None

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: dict, price: float, high: float,
                      tick_data: Optional[dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Check for strategy-specific exit conditions.

        Called BEFORE the engine's standard evaluate_exit. If this returns
        an ExitSignal, it takes priority over engine exits (except EOD/time).

        None = let engine handle exits normally.
        """
        return None

    def update_trailing_stop(self, position: dict, price: float,
                             tick_data: Optional[dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Compute a new trailing stop price.

        None = no trailing stop (engine keeps current stop).
        Returns a float = new stop price (only applied if tighter than current).
        """
        return None

    # ── Indicators ────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        """List of indicators this strategy needs from TickIndicatorEngine.

        Empty list = no indicator engine created (zero overhead).
        Valid values: 'vwap', 'ema', 'opening_range', 'day_high'
        """
        return []

    # ── LLM Prompt Overrides ──────────────────────────────────────────

    def get_llm_system_prompt(self) -> Optional[str]:
        """Override LLM supervisor system prompt. None = use default."""
        return None

    def get_llm_profit_prompt(self) -> Optional[str]:
        """Override LLM profit-taking prompt. None = use default."""
        return None

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

    # ── Tick Callback ─────────────────────────────────────────────────

    def on_tick(self, symbol: str, price: float, size: int = 0,
                timestamp: Optional[datetime] = None):
        """Called on every trade tick (if indicators are required).

        Override to feed indicator engine. Default: no-op.
        """
        pass
