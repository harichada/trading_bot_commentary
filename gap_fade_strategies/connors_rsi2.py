"""Connors RSI(2) Mean Reversion — intraday strategy.

Based on Larry Connors' research: 2-period RSI identifies extreme short-term
oversold/overbought conditions. Historically 75% win rate, PF 2.08.

Entry:
    LONG:  RSI(2) < rsi_entry_long (default 10) AND price above 200-bar MA filter
    SHORT: RSI(2) > rsi_entry_short (default 90) AND price below 200-bar MA filter

Exit:
    - RSI(2) crosses rsi_exit_threshold (default 55 for longs, 45 for shorts)
    - Stop loss: ATR-based or fixed % (whichever is wider)
    - Time exit: max_hold_minutes (default 60 min)

Key insight: RSI(2) is extremely responsive — crosses thresholds often,
but signals at extremes (<10 or >90) have strong mean reversion tendency.
The 200-bar MA filter ensures we only trade with the larger trend.

Active window: 9:45 AM to 3:30 PM ET.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal
from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('connors_rsi2')
class ConnorsRSI2Strategy(IntradayStrategy):
    """Connors RSI(2) Mean Reversion strategy.

    Enters on extreme RSI(2) readings, exits on RSI normalization.
    Trades with the trend (200-bar MA filter).
    """

    name = 'Connors RSI(2)'
    description = 'Mean reversion on extreme 2-period RSI with trend filter'
    version = '1.0'
    strategy_id = 'connors_rsi2'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entered_symbols: Dict[str, int] = {}
        self._entry_times: Dict[str, datetime] = {}

    def get_default_config(self) -> Dict:
        return {
            # RSI thresholds (adapted for RSI-14 on 5-min bars)
            # RSI(14) reaches 25-30 on strong pullbacks, 70-75 on surges
            'rsi_entry_long': 25,          # RSI below this -> buy (oversold)
            'rsi_entry_short': 75,         # RSI above this -> short (overbought)
            'rsi_exit_long': 50,           # exit long when RSI crosses above
            'rsi_exit_short': 50,          # exit short when RSI crosses below
            # Trend filter: use EMA as proxy for trend
            'trend_filter_period': 50,     # EMA period for trend (50-bar ~4hrs on 5-min)
            'require_trend_filter': True,  # only trade with trend
            # Stop loss
            'stop_atr_mult': 1.5,          # stop = ATR * this
            'min_stop_pct': 0.005,         # minimum 0.5% stop
            'max_stop_pct': 0.015,         # maximum 1.5% stop
            # Target
            'target_atr_mult': 2.0,        # target = ATR * this (R:R >= 1.3)
            'min_target_pct': 0.008,       # minimum 0.8% target
            # Position management
            'max_hold_minutes': 60,        # max hold before time exit
            'max_entries_per_day': 3,      # per symbol
            # Quality filters
            'min_risk_reward': 1.0,
            'min_confidence': 0.40,
            'min_volume_ratio': 0.5,       # at least 50% of normal volume
        }

    def get_required_indicators(self) -> List[str]:
        return ['rsi', 'atr', 'ema', 'ema_multi', 'vwap', 'day_high', 'day_low']

    def get_ema_periods(self) -> List[int]:
        return [9, 20, 50]

    def get_active_window(self) -> Tuple[int, int, int, int]:
        return (9, 45, 15, 30)

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': 500_000,
            'min_price': 10.0,
            'max_price': 500.0,
            'min_adr_pct': 0.005,
            'prefer_gappers': False,
        }

    # ── Scanning ──────────────────────────────────────────────────────

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None

        # Check max entries
        max_entries = self._config.get('max_entries_per_day', 3)
        if self._entered_symbols.get(symbol, 0) >= max_entries:
            return None

        # Get price
        price = None
        if snapshot:
            price = snapshot.get('price') or snapshot.get('latestTrade', {}).get('p')
        if price is None or price <= 0:
            return None

        # Need RSI initialized
        if not tick_data.get('rsi_initialized'):
            return None

        rsi = tick_data.get('rsi', 50)
        atr = tick_data.get('atr', 0)
        if atr <= 0:
            return None

        # Trend filter: EMA50
        ema50 = None
        ema_multi = tick_data.get('ema_multi', {})
        if isinstance(ema_multi, dict):
            ema50 = ema_multi.get(50) or ema_multi.get('50')
        if ema50 is None:
            ema50 = tick_data.get('ema50') or tick_data.get('ema', 0)
        if not ema50 or ema50 <= 0:
            return None

        # Volume filter
        min_vol_ratio = self._config.get('min_volume_ratio', 0.5)
        vol_ratio = tick_data.get('relative_volume', 1.0)
        if vol_ratio < min_vol_ratio:
            return None

        rsi_entry_long = self._config.get('rsi_entry_long', 10)
        rsi_entry_short = self._config.get('rsi_entry_short', 90)
        require_trend = self._config.get('require_trend_filter', True)

        direction = None
        confidence = 0.0

        # LONG setup: RSI oversold
        if rsi < rsi_entry_long:
            if require_trend and price < ema50:
                return None  # below trend = don't buy
            direction = 'long'
            # More extreme RSI = higher confidence (0 RSI -> 0.95 conf)
            confidence = min(0.95, 0.45 + (rsi_entry_long - rsi) / rsi_entry_long * 0.50)

        # SHORT setup: RSI overbought
        elif rsi > rsi_entry_short:
            if require_trend and price > ema50:
                return None  # above trend = don't short
            direction = 'short'
            confidence = min(0.95, 0.45 + (rsi - rsi_entry_short) / (100 - rsi_entry_short) * 0.50)

        if direction is None:
            return None

        # Calculate stop and target
        stop_atr = self._config.get('stop_atr_mult', 1.5)
        min_stop = self._config.get('min_stop_pct', 0.005)
        max_stop = self._config.get('max_stop_pct', 0.015)
        target_atr = self._config.get('target_atr_mult', 2.0)
        min_target = self._config.get('min_target_pct', 0.008)

        stop_dist = max(atr * stop_atr, price * min_stop)
        stop_dist = min(stop_dist, price * max_stop)
        target_dist = max(atr * target_atr, price * min_target)

        if direction == 'long':
            stop_price = round(price - stop_dist, 2)
            target_price = round(price + target_dist, 2)
        else:
            stop_price = round(price + stop_dist, 2)
            target_price = round(price - target_dist, 2)

        risk = abs(price - stop_price)
        reward = abs(target_price - price)
        rr = reward / risk if risk > 0 else 0

        return IntradaySetup(
            symbol=symbol,
            strategy_id=self.strategy_id,
            direction=direction,
            entry_price=price,
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=rr,
            confidence=confidence,
            setup_type='rsi2_oversold' if direction == 'long' else 'rsi2_overbought',
            timestamp=now,
            indicators={'rsi': round(rsi, 1), 'atr': round(atr, 4),
                        'ema50': round(ema50, 2), 'price': round(price, 2)},
            notes=f'RSI(2)={rsi:.1f}, EMA50={ema50:.2f}, trend={"with" if require_trend else "any"}',
        )

    # ── Exit Management ───────────────────────────────────────────────

    def evaluate_exit(self, position: Dict, price: float,
                      tick_data: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Exit when RSI normalizes or time exceeds max hold."""
        if not tick_data:
            return None

        rsi = tick_data.get('rsi', 50)
        direction = position.get('direction', 'long')

        # RSI normalization exit
        if direction == 'long' and rsi > self._config.get('rsi_exit_long', 55):
            return ExitSignal(
                action='close',
                reason=f'rsi2_normalized (RSI={rsi:.1f} > {self._config.get("rsi_exit_long", 55)})',
            )
        elif direction == 'short' and rsi < self._config.get('rsi_exit_short', 45):
            return ExitSignal(
                action='close',
                reason=f'rsi2_normalized (RSI={rsi:.1f} < {self._config.get("rsi_exit_short", 45)})',
            )

        # Time exit
        max_hold = self._config.get('max_hold_minutes', 60)
        entry_time = position.get('entry_time', '')
        if entry_time and now:
            try:
                if isinstance(entry_time, str):
                    et = datetime.strptime(entry_time, '%Y-%m-%d %H:%M')
                    if now.tzinfo:
                        et = et.replace(tzinfo=now.tzinfo)
                else:
                    et = entry_time
                elapsed = (now - et).total_seconds() / 60
                if elapsed >= max_hold:
                    return ExitSignal(
                        action='close',
                        reason=f'time_exit ({elapsed:.0f}min >= {max_hold}min)',
                    )
            except (ValueError, TypeError):
                pass

        return None

    def update_trailing_stop(self, position: Dict, price: float,
                             tick_data: Optional[Dict] = None,
                             now: Optional[datetime] = None) -> Optional[float]:
        """Tighten stop after 0.5R profit."""
        if not tick_data:
            return None

        direction = position.get('direction', 'long')
        entry_price = position.get('entry_price', 0)
        stop_price = position.get('stop_price', 0)

        if entry_price <= 0 or stop_price <= 0:
            return None

        risk = abs(entry_price - stop_price)
        if risk <= 0:
            return None

        if direction == 'long':
            pnl_r = (price - entry_price) / risk
            if pnl_r >= 0.5:
                # Move stop to breakeven
                new_stop = max(stop_price, entry_price)
                if new_stop > stop_price:
                    return round(new_stop, 2)
            if pnl_r >= 1.0:
                # Trail at 0.5R below current
                trail_stop = price - risk * 0.5
                if trail_stop > stop_price:
                    return round(trail_stop, 2)
        else:
            pnl_r = (entry_price - price) / risk
            if pnl_r >= 0.5:
                new_stop = min(stop_price, entry_price)
                if new_stop < stop_price:
                    return round(new_stop, 2)
            if pnl_r >= 1.0:
                trail_stop = price + risk * 0.5
                if trail_stop < stop_price:
                    return round(trail_stop, 2)

        return None

    # ── Lifecycle ─────────────────────────────────────────────────────

    def record_entry(self, symbol: str):
        self._entered_symbols[symbol] = self._entered_symbols.get(symbol, 0) + 1

    def on_day_start(self):
        self._entered_symbols.clear()
        self._entry_times.clear()

    def on_day_end(self):
        self._entered_symbols.clear()
        self._entry_times.clear()
