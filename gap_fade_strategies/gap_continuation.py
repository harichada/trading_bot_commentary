"""Gap Continuation — data-driven intraday strategy.

Empirical analysis of 898 gap-down events (>4%, 2024-2025):
- 55.5% close BELOW the open (continuation)
- Average P&L if shorted at open: +1.32%
- Profit Factor (hold to close): 1.61
- 66% bounce >1.5% intraday — tight stops get DESTROYED

The key insight: after a big gap down, the stock WILL bounce (dead cat bounce),
but it usually closes lower than the open. Tight stops get swept by the bounce.
The edge comes from HOLDING THROUGH the bounce.

Strategy:
    1. Detect big gap-down stocks (>4%) at market open
    2. Wait for initial volatility to settle (enter at 10:00 AM)
    3. Short with WIDE stop (2.5%) or no stop
    4. Hold until 3:00 PM (let the selling resume in the afternoon)
    5. Position size at 50% normal (wider stop = smaller size)

This is NOT a scalp. It's a conviction trade based on institutional selling
that takes all day to play out. The bounce is noise; the close is signal.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('gap_continuation')
class GapContinuationStrategy(IntradayStrategy):
    """Gap Continuation — data-driven short on big gap-downs.

    Based on analysis of 898 events: 55.5% continuation, PF 1.61.
    Uses wide stops to survive the intraday dead-cat bounce.
    """

    name = 'Gap Continuation'
    description = 'Short big gap-downs with wide stops (data-driven edge)'
    version = '2.0'
    strategy_id = 'gap_continuation'

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
            'min_price': 5.0,              # avoid penny stocks
            # Daily RSI filter (data: RSI<30 bounces, RSI>50 continues)
            'min_daily_rsi': 30,           # skip oversold stocks (they bounce)
            # Entry timing
            'earliest_entry_bar': 6,       # ~10:00 AM (bar 6 = 30 min after open)
            # Stop (WIDE — survive the dead cat bounce)
            'stop_pct': 0.025,             # 2.5% stop — wider than normal
            'min_stop_pct': 0.015,         # minimum 1.5% stop
            # Target
            'target_pct': 0.015,           # 1.5% target (data shows 74.5% reach this)
            'target_rr': 0.6,             # target = 0.6R (tighter than stop)
            # Time management
            'exit_hour': 15,              # 3:00 PM exit (afternoon selling complete)
            'exit_min': 0,
            # Position management
            'max_entries_per_day': 1,      # ONE entry per symbol
            # Quality
            'min_risk_reward': 0.4,        # Allow asymmetric (wider stop, tighter target)
            'min_confidence': 0.40,
        }

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema', 'ema_multi', 'rsi', 'atr', 'day_high', 'day_low',
                'bar_history', 'volume_profile']

    def get_ema_periods(self) -> List[int]:
        return [9, 20]

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (9, 55, 11, 0)  # Narrow entry window: 9:55-11:00 AM

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

        # Need enough bars (wait for initial volatility)
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
        ema9 = tick_data.get('ema9', 0)
        ema20 = tick_data.get('ema20', 0)
        vwap = tick_data.get('vwap', 0)

        if not tick_data.get('rsi_initialized') or atr <= 0:
            return None

        # Daily RSI context (used for confidence, not filtering)
        # Data on daily bars showed RSI<30 bounces, but on liquid minute-bar
        # stocks the effect is weaker — so we use RSI for confidence only
        daily_rsi = self._daily_rsi.get(symbol, 50)

        # Intraday RSI check: don't short extremely oversold (bounce likely)
        if rsi < 20:
            return None  # too oversold, bounce probable

        # Calculate stop and target
        stop_pct = self._config.get('stop_pct', 0.025)
        target_pct = self._config.get('target_pct', 0.015)

        # Use ATR to adjust stop if available
        if atr > 0:
            atr_stop = atr * 2.0 / price  # 2 ATR as % of price
            stop_pct = max(stop_pct, atr_stop)

        min_stop = self._config.get('min_stop_pct', 0.015)
        stop_pct = max(stop_pct, min_stop)

        stop_price = round(price * (1 + stop_pct), 2)
        target_price = round(price * (1 - target_pct), 2)

        risk = abs(price - stop_price)
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0

        # Confidence based on gap size, daily RSI, and intraday context
        confidence = 0.50
        if abs(gap_pct) >= 0.10:
            confidence += 0.15  # huge gap = very bearish
        elif abs(gap_pct) >= 0.06:
            confidence += 0.10
        # Daily RSI: higher = more room to fall (better for shorts)
        if daily_rsi > 50:
            confidence += 0.10  # fresh gap from strength = panic selling
        elif daily_rsi > 40:
            confidence += 0.05
        if rsi < 40:
            confidence += 0.05  # intraday already weak
        if ema9 > 0 and ema20 > 0 and ema9 < ema20:
            confidence += 0.05  # EMAs confirm
        if vwap > 0 and price < vwap:
            confidence += 0.05  # below VWAP
        confidence = min(confidence, 0.90)

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction='short',
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=rr,
            confidence=confidence,
            setup_type='gap_down_short',
            timestamp=now,
            indicators={
                'gap_pct': round(gap_pct * 100, 2),
                'daily_rsi': round(daily_rsi, 1),
                'rsi': round(rsi, 1),
                'vwap': round(vwap, 2) if vwap else 0,
                'atr': round(atr, 4),
            },
            notes=f'Gap={gap_pct:+.1%} DailyRSI={daily_rsi:.0f} RSI={rsi:.0f} stop={stop_pct:.1%} target={target_pct:.1%}',
        )

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Time-based exit at 3:00 PM.

        The data shows gap-down selling resumes in the afternoon.
        By 3:00 PM, most of the move is done. Exit here.
        """
        if not now:
            return None

        exit_h = self._config.get('exit_hour', 15)
        exit_m = self._config.get('exit_min', 0)
        if now.hour > exit_h or (now.hour == exit_h and now.minute >= exit_m):
            return ExitSignal(
                action='close',
                reason=f'time_exit_3pm',
            )

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """No trailing stop — the wide stop and time exit handle everything.

        Trailing stops on gap-down trades get triggered by the dead cat bounce.
        """
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
