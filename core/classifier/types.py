"""Contract types for the side-classifier subsystem.

v-side-classifier-types-2026-05-13. See
``.claude/PLAN_symbol_side_classifier.md`` for the design these types
encode. Rule-based and trained-model implementations agree on these
shapes; the composer (``core/classifier/composer.py``) consumes both.

The decision space is two sides only — `LONG` and `SHORT`. "Don't
trade" is represented by an empty ``allowed_sides`` frozenset, not by
a third enum value. That keeps the set algebra natural:
``final_allowed = rule_allowed & trained_allowed``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Optional


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


@dataclass(frozen=True)
class SymbolSideDecision:
    """A single per-symbol decision. Immutable so it can be cached
    + audited without defensive copies. ``feature_dump`` carries the
    inputs the decision was made on, for post-mortem review."""

    symbol: str
    allowed_sides: FrozenSet[Side]
    long_score: float
    short_score: float
    primary_reason: str
    feature_dump: dict

    def allows_long(self) -> bool:
        return Side.LONG in self.allowed_sides

    def allows_short(self) -> bool:
        return Side.SHORT in self.allowed_sides

    def is_forbidden(self) -> bool:
        """Empty allowed_sides — symbol should not be traded today."""
        return len(self.allowed_sides) == 0


@dataclass(frozen=True)
class SymbolFeatures:
    """Inputs to the rule-based classifier. Field set mirrors PLAN §4.

    Defaults are deliberately *neutral* — i.e., values that would not
    push the classifier toward either side. That makes test fixtures
    short (set only the field(s) under test); it also means missing
    feature data degrades gracefully to "no edge either way" rather
    than to a confident wrong answer.
    """

    symbol: str
    close: float

    # ── 4.1 Trend ────────────────────────────────────────────────────
    sma_20_daily: Optional[float] = None
    sma_50_daily: Optional[float] = None
    intraday_trend_slope: float = 0.0     # regression slope of 5-min closes
    lower_lows: int = 0                   # consecutive lower-low daily bars
    higher_highs: int = 0                 # consecutive higher-high daily bars

    # ── 4.2 Range position ──────────────────────────────────────────
    daily_atr_pct: Optional[float] = None # ATR_14 / close, daily
    pct_from_52w_high: Optional[float] = None
    pct_from_52w_low: Optional[float] = None

    # Indicator from the strategies' existing computation; useful for
    # range-position scoring and aligning with strategy entry logic.
    rsi_14: Optional[float] = None

    # ── 4.3 Volume ──────────────────────────────────────────────────
    vol_ratio_today: float = 1.0          # today_volume / avg_volume_20d
    up_vs_down_volume_5d: float = 1.0     # >1 = accumulation, <1 = distribution
    avg_daily_volume_20d: Optional[float] = None

    # ── 4.4 News flow ───────────────────────────────────────────────
    news_sentiment_7d: float = 0.0        # in [-1, 1]
    news_fresh_count_today: int = 0
    news_recency_min: Optional[float] = None

    # ── 4.5 Relative strength ───────────────────────────────────────
    rs_vs_spy_5d: float = 0.0             # symbol_return - SPY_return, 5d cum
    rs_vs_sector_5d: float = 0.0

    # ── 4.6 Hard-gate events ────────────────────────────────────────
    earnings_within_2d: bool = False
    ex_dividend_within_1d: bool = False
    halt_today: bool = False
    hard_to_borrow: bool = False

    # ── Free-form extras (sector ID, regime tag, etc.) for the
    #   trained model's static features. Rule-based ignores these.
    extras: dict = field(default_factory=dict)
