"""Gap Bounce — buy oversold gap-downs for the dead cat bounce.

Empirical analysis (2024-2025, daily_bars):
    Gap-down 6-10% with daily RSI(14) < 35:
        - 60.0% close ABOVE the open (bounce)
        - Average P&L if bought at open: +2.09%
        - Profit Factor: 2.25
        - Average max bounce (high - open): 6.46%
        - Average max drawdown (open - low): 3.53%

    Gap-down 4-6% with daily RSI(14) < 35:
        - 54.3% close ABOVE the open
        - Average P&L: +0.95%, PF 1.53

The key insight: an already oversold stock gapping down further is hitting
exhaustion selling. Short covering and bottom fishing create a bounce.
This is the OPPOSITE of gap_continuation (which shorts fresh gap-downs).

Strategy:
    1. Detect gap-down stocks (>4%) where daily RSI(14) was ALREADY < 40
    2. Wait for initial selling to exhaust (enter after bar 6 = ~10:00 AM)
    3. Buy LONG — NO stop, NO target (time-only management)
    4. Exit at 3:00 PM (the bounce is gradual; stops/targets hurt this trade)
    5. The edge comes from holding through the noise
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('gap_bounce')
class GapBounceStrategy(IntradayStrategy):
    """Gap Bounce — buy oversold gap-downs for dead cat bounce.

    Based on analysis: gap-down 6-10% + RSI<35 = 60% bounce, PF 2.25.
    Uses wide stops to survive any further selling.
    """

    name = 'Gap Bounce'
    description = 'Long oversold gap-downs for bounce (data-driven edge)'
    version = '1.0'
    strategy_id = 'gap_bounce'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}
        self._prev_close: Dict[str, float] = {}  # fed by backtester
        self._daily_rsi: Dict[str, float] = {}    # fed by backtester
        self._day_gaps: Dict[str, float] = {}
        self._entered_today: Dict[str, bool] = {}

    def get_default_config(self) -> Dict:
        return {
            # Gap filter
            'min_gap_down_pct': 0.04,      # 4% gap down minimum
            'preferred_gap_min': 0.06,     # 6%+ is the sweet spot
            'min_price': 5.0,
            # Daily RSI filter (pre-gap)
            'max_daily_rsi': 40,           # Only buy when daily RSI oversold
            # Entry timing
            'earliest_entry_bar': 6,       # ~10:00 AM (let opening chaos settle)
            # NO stop/target — time exit handles everything
            # Data shows stops/targets both hurt this trade
            'stop_pct': 0.50,              # Effectively no stop
            'min_stop_pct': 0.40,
            'target_pct': 0.50,            # Effectively no target
            # Time management (the ONLY exit mechanism)
            'exit_hour': 15,               # 3:00 PM exit
            'exit_min': 0,
            # Position management
            'max_entries_per_day': 1,
            # Quality
            'min_risk_reward': 0.01,       # No meaningful R:R with time-only
            'min_confidence': 0.40,
        }

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr', 'day_high', 'day_low',
                'bar_history', 'volume_profile']

    def get_ema_periods(self) -> List[int]:
        return [9, 20]

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (9, 55, 11, 0)  # Narrow window: 9:55-11:00 AM

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 5.0,
            'max_price': 500.0,
            'min_adr_pct': 0.005,
            'prefer_gappers': True,
        }

    def _detect_gap(self, symbol: str, tick_data: Dict) -> Optional[float]:
        if symbol in self._day_gaps:
            return self._day_gaps[symbol]

        bar_history = tick_data.get('bar_history', [])
        if not bar_history:
            return None

        first_bar = bar_history[0]
        day_open = getattr(first_bar, 'open', None)
        if day_open is None and isinstance(first_bar, dict):
            day_open = first_bar.get('open')
        if not day_open or day_open <= 0:
            return None

        prev_close = self._prev_close.get(symbol)
        if not prev_close or prev_close <= 0:
            return None

        gap_pct = (day_open - prev_close) / prev_close
        self._day_gaps[symbol] = gap_pct
        return gap_pct

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None

        # One entry per symbol per day
        if self._entered_today.get(symbol):
            return None

        # Need enough bars
        min_bars = self._config.get('earliest_entry_bar', 6)
        bar_count = tick_data.get('bar_count', 0)
        if bar_count < min_bars:
            return None

        # Detect gap
        gap_pct = self._detect_gap(symbol, tick_data)
        if gap_pct is None:
            return None

        min_gap = self._config.get('min_gap_down_pct', 0.04)
        if gap_pct > -min_gap:
            return None  # not a big enough gap down

        # CRITICAL FILTER: daily RSI must be oversold (< 35)
        daily_rsi = self._daily_rsi.get(symbol, 50)
        max_rsi = self._config.get('max_daily_rsi', 35)
        if daily_rsi > max_rsi:
            return None  # not oversold enough — let gap_continuation handle this

        # Get price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None or price <= 0:
            return None

        min_price = self._config.get('min_price', 5.0)
        if price < min_price:
            return None

        # Basic indicators
        atr = tick_data.get('atr', 0)
        rsi = tick_data.get('rsi', 50)
        vwap = tick_data.get('vwap', 0)

        if not tick_data.get('rsi_initialized') or atr <= 0:
            return None

        # Don't buy if intraday RSI is already overbought (bounce already happened)
        if rsi > 70:
            return None

        # Calculate stop and target
        stop_pct = self._config.get('stop_pct', 0.035)
        target_pct = self._config.get('target_pct', 0.020)

        # Use ATR to adjust stop
        if atr > 0:
            atr_stop = atr * 2.5 / price  # 2.5 ATR as % of price
            stop_pct = max(stop_pct, atr_stop)

        min_stop = self._config.get('min_stop_pct', 0.020)
        stop_pct = max(stop_pct, min_stop)

        stop_price = round(price * (1 - stop_pct), 2)
        target_price = round(price * (1 + target_pct), 2)

        risk = abs(price - stop_price)
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0

        # Confidence based on gap size, daily RSI, and intraday context
        confidence = 0.50
        preferred_gap = self._config.get('preferred_gap_min', 0.06)
        if abs(gap_pct) >= 0.10:
            confidence += 0.15  # huge gap = high bounce probability
        elif abs(gap_pct) >= preferred_gap:
            confidence += 0.10  # sweet spot (6-10% gap)
        if daily_rsi < 25:
            confidence += 0.10  # deeply oversold = stronger bounce
        elif daily_rsi < 30:
            confidence += 0.05
        if vwap > 0 and price > vwap:
            confidence += 0.05  # already bouncing above VWAP
        if rsi > 30 and rsi < 50:
            confidence += 0.05  # intraday selling exhausting
        confidence = min(confidence, 0.90)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction='long',
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=rr,
            confidence=confidence,
            setup_type='gap_bounce_long',
            timestamp=now,
            indicators={
                'gap_pct': round(gap_pct * 100, 2),
                'daily_rsi': round(daily_rsi, 1),
                'intraday_rsi': round(rsi, 1),
                'vwap': round(vwap, 2) if vwap else 0,
                'atr': round(atr, 4),
            },
            notes=f'Gap={gap_pct:+.1%} DailyRSI={daily_rsi:.0f} stop={stop_pct:.1%} target={target_pct:.1%}',
        )

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Time-based exit at 3:00 PM."""
        if not now:
            return None

        exit_h = self._config.get('exit_hour', 15)
        exit_m = self._config.get('exit_min', 0)
        if now.hour > exit_h or (now.hour == exit_h and now.minute >= exit_m):
            return ExitSignal(
                action='close',
                reason='time_exit_3pm',
            )

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """No trailing stop — wide stop + time exit handle everything."""
        return None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1
        self._entered_today[symbol] = True

    def on_day_start(self):
        self._entered_symbols.clear()
        self._day_gaps.clear()
        self._entered_today.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
        self._day_gaps.clear()
        self._entered_today.clear()
