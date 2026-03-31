"""Aziz ABCD + VWAP Composite Strategy.

Based on Andrew Aziz's "Advanced Techniques in Day Trading":
- Waits for initial move (A), pullback to VWAP/EMA (B→C), breakout (D)
- Requires price alignment with VWAP and EMA trend
- 1.5:1 minimum risk/reward
- Stops below consolidation, targets prior extremes
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('aziz_abcd_vwap')
class AzizABCDVWAPStrategy(IntradayStrategy):

    name = 'Aziz ABCD + VWAP'
    description = 'ABCD pullback-to-VWAP breakout with EMA trend confirmation'
    version = '2.0'
    strategy_id = 'aziz_abcd_vwap'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entries_today: Dict[str, int] = {}
        # Per-symbol bar history for pattern detection
        self._bar_history: Dict[str, list] = {}  # symbol -> [(price, vwap, ema9, ema20, volume, ts)]

    @staticmethod
    def get_default_config() -> Dict:
        return {
            'earliest_entry_min': 5,
            'latest_entry_hour': 14, 'latest_entry_min': 30,
            'exit_hour': 15, 'exit_min': 0,
            'min_risk_reward': 1.5,
            'max_entries_per_day': 3,
            'min_price': 5.0, 'max_price': 500.0,
            'min_atr_pct': 0.005,
            # Pullback detection
            'lookback_bars': 20,        # look back N bars for swing high/low
            'pullback_min_pct': 0.002,  # price must pull back at least 0.2%
            'pullback_max_pct': 0.015,  # but not more than 1.5% (that's a reversal, not pullback)
            # Breakout
            'breakout_above_ema9': True,
            'price_above_vwap_for_long': True,
            'ema9_above_ema20_for_long': True,
            'min_confidence': 0.3,
        }

    def get_required_indicators(self) -> list:
        return ['vwap', 'ema9', 'ema20', 'rsi', 'atr', 'volume_surge_ratio']

    def get_active_window(self) -> Tuple[int, int, int, int]:
        c = self._config
        return (9, c.get('earliest_entry_min', 5) + 30, c.get('latest_entry_hour', 14), c.get('latest_entry_min', 30))

    def get_watchlist_criteria(self) -> Dict:
        return {'min_volume': 200000, 'min_price': self._config.get('min_price', 5.0),
                'max_price': self._config.get('max_price', 500.0), 'prefer_gappers': True}

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None
        cfg = self._config

        # Timing
        mkt_min = (now.hour - 9) * 60 + (now.minute - 30)
        if mkt_min < cfg.get('earliest_entry_min', 5):
            return None
        if now.hour > cfg.get('latest_entry_hour', 14):
            return None
        if now.hour == cfg.get('latest_entry_hour', 14) and now.minute > cfg.get('latest_entry_min', 30):
            return None
        if self._entries_today.get(symbol, 0) >= cfg.get('max_entries_per_day', 3):
            return None

        price = snapshot.get('price', 0) if snapshot else tick_data.get('price', 0)
        if price <= 0 or price < cfg.get('min_price', 5) or price > cfg.get('max_price', 500):
            return None

        vwap = tick_data.get('vwap', 0)
        ema9 = tick_data.get('ema9', tick_data.get('ema_9', 0))
        ema20 = tick_data.get('ema20', tick_data.get('ema_21', 0))
        atr = tick_data.get('atr', 0)
        rsi = tick_data.get('rsi', 50)
        vol_surge = tick_data.get('volume_surge_ratio', tick_data.get('volume_surge', 1.0))
        volume = snapshot.get('volume', 0) if snapshot else 0

        if vwap <= 0 or atr <= 0:
            return None
        if atr / price < cfg.get('min_atr_pct', 0.005):
            return None

        # Track bar history
        hist = self._bar_history.setdefault(symbol, [])
        hist.append({'p': price, 'v': vwap, 'e9': ema9, 'e20': ema20, 'vol': volume, 'ts': now})
        if len(hist) > 60:
            hist.pop(0)

        lookback = cfg.get('lookback_bars', 20)
        if len(hist) < lookback:
            return None

        recent = hist[-lookback:]
        prices = [b['p'] for b in recent]
        recent_high = max(prices)
        recent_low = min(prices)
        high_idx = prices.index(recent_high)
        low_idx = prices.index(recent_low)

        # ── LONG SETUP: swing high (A), pullback (B→C), current bar breaks above ──
        if high_idx < len(prices) - 3:  # high was at least 3 bars ago
            pullback_depth = (recent_high - price) / recent_high
            has_pulled_back = cfg.get('pullback_min_pct', 0.002) <= pullback_depth <= cfg.get('pullback_max_pct', 0.015)

            # Price recovering — current bar above ema9, ema9 > ema20, price near VWAP or above
            ema_bullish = ema9 > ema20 if cfg.get('ema9_above_ema20_for_long', True) else True
            above_vwap = price > vwap if cfg.get('price_above_vwap_for_long', True) else True
            above_ema9 = price > ema9 if cfg.get('breakout_above_ema9', True) else True

            # Check if price just crossed above ema9 (breakout from pullback)
            prev_price = prices[-2] if len(prices) >= 2 else price
            crossed_ema9 = prev_price <= ema9 and price > ema9

            if has_pulled_back and ema_bullish and above_vwap and (above_ema9 or crossed_ema9):
                # Calculate stop and target
                stop_price = min(prices[-5:]) - atr * 0.3  # below recent low + buffer
                risk = price - stop_price
                if risk <= 0 or risk > price * 0.02:  # max 2% risk
                    return None

                target_price = recent_high  # target = prior swing high
                reward = target_price - price
                rr = reward / risk if risk > 0 else 0

                if rr >= cfg.get('min_risk_reward', 1.5):
                    self._entries_today[symbol] = self._entries_today.get(symbol, 0) + 1
                    confidence = min(0.9, 0.4 + (rr - 1.5) * 0.1 + (0.1 if crossed_ema9 else 0) + (0.1 if vol_surge > 1.2 else 0))

                    return IntradaySetup(
                        symbol=symbol, strategy_id=self.strategy_id, direction='long',
                        entry_price=price, stop_price=round(stop_price, 2),
                        target_price=round(target_price, 2), risk_reward=round(rr, 2),
                        confidence=round(confidence, 2), setup_type='abcd_long',
                        indicators={'vwap': round(vwap, 2), 'ema9': round(ema9, 2), 'ema20': round(ema20, 2),
                                    'rsi': round(rsi, 1), 'vol_surge': round(vol_surge, 2), 'pullback': round(pullback_depth * 100, 2)},
                        timestamp=now,
                        notes=f'Pullback {pullback_depth*100:.1f}% from ${recent_high:.2f}, RR={rr:.1f}',
                    )

        # ── SHORT SETUP: swing low (A), bounce (B→C), current bar breaks below ──
        if low_idx < len(prices) - 3:
            bounce_depth = (price - recent_low) / recent_low if recent_low > 0 else 0
            has_bounced = cfg.get('pullback_min_pct', 0.002) <= bounce_depth <= cfg.get('pullback_max_pct', 0.015)

            ema_bearish = ema9 < ema20
            below_vwap = price < vwap
            below_ema9 = price < ema9

            prev_price = prices[-2] if len(prices) >= 2 else price
            crossed_below_ema9 = prev_price >= ema9 and price < ema9

            if has_bounced and ema_bearish and below_vwap and (below_ema9 or crossed_below_ema9):
                stop_price = max(prices[-5:]) + atr * 0.3
                risk = stop_price - price
                if risk <= 0 or risk > price * 0.02:
                    return None

                target_price = recent_low
                reward = price - target_price
                rr = reward / risk if risk > 0 else 0

                if rr >= cfg.get('min_risk_reward', 1.5):
                    self._entries_today[symbol] = self._entries_today.get(symbol, 0) + 1
                    confidence = min(0.9, 0.4 + (rr - 1.5) * 0.1 + (0.1 if crossed_below_ema9 else 0) + (0.1 if vol_surge > 1.2 else 0))

                    return IntradaySetup(
                        symbol=symbol, strategy_id=self.strategy_id, direction='short',
                        entry_price=price, stop_price=round(stop_price, 2),
                        target_price=round(target_price, 2), risk_reward=round(rr, 2),
                        confidence=round(confidence, 2), setup_type='abcd_short',
                        indicators={'vwap': round(vwap, 2), 'ema9': round(ema9, 2), 'ema20': round(ema20, 2),
                                    'rsi': round(rsi, 1), 'vol_surge': round(vol_surge, 2), 'bounce': round(bounce_depth * 100, 2)},
                        timestamp=now,
                        notes=f'Bounce {bounce_depth*100:.1f}% from ${recent_low:.2f}, RR={rr:.1f}',
                    )

        return None

    def evaluate_exit(self, position: Dict, tick_data: Dict,
                      current_price: float, now: datetime) -> Optional[Tuple[bool, str]]:
        cfg = self._config
        if now.hour >= cfg.get('exit_hour', 15) and now.minute >= cfg.get('exit_min', 0):
            return (True, 'time_exit')
        return None

    def validate_setup(self, setup: IntradaySetup, tick_data: Dict,
                       now: datetime) -> Tuple[bool, str]:
        if setup.risk_reward < self._config.get('min_risk_reward', 1.5):
            return (False, f'R:R {setup.risk_reward:.1f} too low')
        return (True, 'valid')

    def reset_daily(self):
        self._entries_today.clear()
        self._bar_history.clear()
