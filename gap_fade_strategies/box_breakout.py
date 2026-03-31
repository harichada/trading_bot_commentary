"""Box Breakout / Range Breakout Strategy.

Identifies stocks consolidating in a tight sideways range ("box") on 15-min bars,
enters on confirmed breakout (candle close outside box) with volume confirmation.
Stop on opposite side of box, target = N x box height.

Runs AFTER gap fade window: 10:30 AM - 3:30 PM ET.
Complements gap fade — uses the same scanner universe but trades momentum/continuation
instead of mean reversion. Zero overlap in timing.

Candidate sourcing:
- Gap scanner candidates that weren't traded (filtered out by gap fade criteria)
- Stocks that were traded and closed (re-entry on box breakout)
- High volume movers not in gap scan
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .intraday_base import IntradaySetup, IntradayStrategy
from .intraday_registry import IntradayStrategyRegistry

logger = logging.getLogger(__name__)


@IntradayStrategyRegistry.register('box_breakout')
class BoxBreakoutStrategy(IntradayStrategy):
    """Consolidation range breakout on 15-minute bars.

    Detects tight sideways ranges, enters when price closes outside
    with volume surge. Stop on opposite side, target at N x box height.
    """

    name = 'Box Breakout'
    description = 'Consolidation range breakout with volume confirmation on 15-min bars'
    version = '1.0'
    strategy_id = 'box_breakout'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._entries_today: Dict[str, int] = {}
        self._bar_history: Dict[str, List[Dict]] = {}  # symbol -> list of 15-min bars
        self._last_bar_time: Dict[str, datetime] = {}   # track when we last built a 15-min bar
        self._current_15m: Dict[str, Dict] = {}          # accumulating current 15-min bar from 1-min bars
        self._detected_boxes: Dict[str, Dict] = {}       # symbol -> box info if detected
        self._total_positions = 0

    @staticmethod
    def get_default_config() -> Dict:
        return {
            # Timing
            'start_hour': 10, 'start_min': 30,
            'end_hour': 15, 'end_min': 30,
            'eod_exit_hour': 15, 'eod_exit_min': 50,

            # Box detection
            'timeframe_minutes': 15,
            'min_consolidation_bars': 4,
            'max_consolidation_bars': 20,
            'min_box_height_pct': 0.003,    # 0.3% of price
            'max_box_height_pct': 0.015,    # 1.5% of price

            # Breakout confirmation
            'breakout_confirmation': 'close',  # 'close' or 'wick'
            'volume_multiplier': 1.5,

            # Risk / Reward
            'risk_reward_ratio': 2.0,
            'stop_buffer_pct': 0.05,        # 5% of box height added as buffer
            'move_stop_to_breakeven': True,  # after 1x box height profit
            'partial_exit_at_1x': True,      # sell 50% at 1x box height
            'partial_exit_pct': 0.5,

            # Position management
            'max_positions': 3,
            'max_entries_per_symbol': 2,
            'position_size_risk_pct': 0.02,  # 2% account risk per trade

            # Filters
            'min_price': 5.0,
            'max_price': 500.0,
            'min_avg_volume': 500000,
            'atr_filter': True,              # require ATR > box height
            'use_gap_scanner_universe': True,

            # Confidence
            'min_confidence': 0.4,
        }

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'ema9', 'ema20', 'atr', 'volume_surge_ratio', 'rsi']

    def get_active_window(self) -> Tuple[int, int, int, int]:
        c = self._config
        return (c.get('start_hour', 10), c.get('start_min', 30),
                c.get('end_hour', 15), c.get('end_min', 30))

    def get_watchlist_criteria(self) -> Dict:
        return {
            'min_volume': self._config.get('min_avg_volume', 500000),
            'min_price': self._config.get('min_price', 5.0),
            'max_price': self._config.get('max_price', 500.0),
            'prefer_gappers': self._config.get('use_gap_scanner_universe', True),
        }

    def _aggregate_to_15m(self, symbol: str, price: float, high: float, low: float,
                           volume: int, now: datetime) -> Optional[Dict]:
        """Aggregate 1-min bars into 15-min bars. Returns completed 15-min bar or None."""
        cfg = self._config
        tf = cfg.get('timeframe_minutes', 15)

        # Determine which 15-min window this bar belongs to
        total_minutes = now.hour * 60 + now.minute
        window_start = (total_minutes // tf) * tf

        cur = self._current_15m.get(symbol)

        if cur is None or cur.get('window') != window_start:
            # New 15-min window — finalize previous bar if exists
            completed = None
            if cur and cur.get('count', 0) > 0:
                completed = {
                    'open': cur['open'], 'high': cur['high'],
                    'low': cur['low'], 'close': cur['close'],
                    'volume': cur['volume'], 'ts': cur['ts'],
                    'count': cur['count'],
                }

            # Start new bar
            self._current_15m[symbol] = {
                'window': window_start,
                'open': price, 'high': high, 'low': low, 'close': price,
                'volume': volume, 'ts': now, 'count': 1,
            }
            return completed
        else:
            # Update current bar
            cur['high'] = max(cur['high'], high)
            cur['low'] = min(cur['low'], low)
            cur['close'] = price
            cur['volume'] = cur.get('volume', 0) + volume
            cur['count'] = cur.get('count', 0) + 1
            return None

    def _detect_box(self, symbol: str) -> Optional[Dict]:
        """Scan bar history for a consolidation box.

        Returns box info dict or None if no valid box found.
        """
        cfg = self._config
        bars = self._bar_history.get(symbol, [])
        min_bars = cfg.get('min_consolidation_bars', 4)
        max_bars = cfg.get('max_consolidation_bars', 20)

        if len(bars) < min_bars:
            return None

        # Try different window sizes from min_bars to max_bars
        for window_size in range(min_bars, min(max_bars + 1, len(bars) + 1)):
            recent = bars[-window_size:]
            box_high = max(b['high'] for b in recent)
            box_low = min(b['low'] for b in recent)
            mid = (box_high + box_low) / 2

            if mid <= 0:
                continue

            box_height = box_high - box_low
            box_height_pct = box_height / mid

            # Check box height is within acceptable range
            if box_height_pct < cfg.get('min_box_height_pct', 0.003):
                continue
            if box_height_pct > cfg.get('max_box_height_pct', 0.015):
                continue

            # Verify ALL bars are contained within the box
            all_contained = all(
                b['high'] <= box_high * 1.001 and b['low'] >= box_low * 0.999
                for b in recent
            )

            if all_contained:
                avg_vol = sum(b['volume'] for b in recent) / len(recent) if recent else 0
                return {
                    'high': box_high,
                    'low': box_low,
                    'height': box_height,
                    'height_pct': box_height_pct,
                    'bars': window_size,
                    'avg_volume': avg_vol,
                    'mid': mid,
                }

        return None

    def scan_for_setups(self, symbol: str, tick_data: Dict,
                        snapshot: Optional[Dict],
                        now: datetime) -> Optional[IntradaySetup]:
        if not tick_data:
            return None

        cfg = self._config

        # ── Timing gate ──
        if now.hour < cfg.get('start_hour', 10):
            return None
        if now.hour == cfg.get('start_hour', 10) and now.minute < cfg.get('start_min', 30):
            return None
        if now.hour > cfg.get('end_hour', 15):
            return None
        if now.hour == cfg.get('end_hour', 15) and now.minute > cfg.get('end_min', 30):
            return None

        # ── Entry limits ──
        if self._entries_today.get(symbol, 0) >= cfg.get('max_entries_per_symbol', 2):
            return None
        if self._total_positions >= cfg.get('max_positions', 3):
            return None

        # ── Price ──
        price = snapshot.get('price', 0) if snapshot else tick_data.get('price', 0)
        if price <= 0:
            return None
        if price < cfg.get('min_price', 5) or price > cfg.get('max_price', 500):
            return None

        volume = snapshot.get('volume', 0) if snapshot else 0
        atr = tick_data.get('atr', 0)
        high = tick_data.get('high_of_day', price)
        low = tick_data.get('low_of_day', price)
        vwap = tick_data.get('vwap', 0)

        # ── Aggregate 1-min bars into 15-min bars ──
        bar_high = snapshot.get('high', price) if snapshot else price
        bar_low = snapshot.get('low', price) if snapshot else price
        completed_bar = self._aggregate_to_15m(symbol, price, bar_high, bar_low, volume, now)

        if completed_bar:
            # Add completed 15-min bar to history
            hist = self._bar_history.setdefault(symbol, [])
            hist.append(completed_bar)
            if len(hist) > 30:
                hist.pop(0)

            # Detect box on the history (excluding the just-completed bar)
            box = self._detect_box(symbol)
            if box:
                self._detected_boxes[symbol] = box

        # ── Check for BREAKOUT against any detected box ──
        box = self._detected_boxes.get(symbol)
        if not box:
            return None

        avg_vol = box.get('avg_volume', 1)
        min_vol_mult = cfg.get('volume_multiplier', 1.5)

        # Volume surge: compare current bar or completed 15m bar vs box average
        if completed_bar:
            vol_surge = completed_bar['volume'] / avg_vol if avg_vol > 0 else 0
            check_price = completed_bar['close']
        else:
            # Per-minute volume scaled to 15-min equivalent
            vol_surge = (volume * cfg.get('timeframe_minutes', 15)) / avg_vol if avg_vol > 0 else 0
            check_price = price

        # ATR filter
        if cfg.get('atr_filter', True) and atr > 0:
            if atr < box['height'] * 0.5:
                return None

        box_height = box['height']
        buffer = box_height * cfg.get('stop_buffer_pct', 0.05)

        # ── LONG BREAKOUT ──
        if check_price > box['high'] and vol_surge >= min_vol_mult:
            entry_price = check_price
            stop_price = box['low'] - buffer
            risk = entry_price - stop_price
            if risk <= 0 or risk > price * 0.03:
                return None

            rr = cfg.get('risk_reward_ratio', 2.0)
            target_price = entry_price + (box_height * rr)
            actual_rr = (target_price - entry_price) / risk
            confidence = self._calc_confidence(box, vol_surge, vwap, price, 'long')
            if confidence < cfg.get('min_confidence', 0.4):
                return None

            self._entries_today[symbol] = self._entries_today.get(symbol, 0) + 1
            self._total_positions += 1
            del self._detected_boxes[symbol]  # consumed

            return IntradaySetup(
                symbol=symbol, strategy_id=self.strategy_id, direction='long',
                entry_price=round(entry_price, 2), stop_price=round(stop_price, 2),
                target_price=round(target_price, 2), risk_reward=round(actual_rr, 2),
                confidence=round(confidence, 2), setup_type='box_breakout_long',
                indicators={'box_high': round(box['high'], 2), 'box_low': round(box['low'], 2),
                            'box_height_pct': round(box['height_pct'] * 100, 2),
                            'box_bars': box['bars'], 'vol_surge': round(vol_surge, 2),
                            'vwap': round(vwap, 2), 'atr': round(atr, 4)},
                timestamp=now,
                notes=f'Box [{box["low"]:.2f}-{box["high"]:.2f}] {box["bars"]} bars, vol {vol_surge:.1f}x',
            )

        # ── SHORT BREAKOUT ──
        if check_price < box['low'] and vol_surge >= min_vol_mult:
            entry_price = check_price
            stop_price = box['high'] + buffer
            risk = stop_price - entry_price
            if risk <= 0 or risk > price * 0.03:
                return None

            rr = cfg.get('risk_reward_ratio', 2.0)
            target_price = entry_price - (box_height * rr)
            actual_rr = (entry_price - target_price) / risk
            confidence = self._calc_confidence(box, vol_surge, vwap, price, 'short')
            if confidence < cfg.get('min_confidence', 0.4):
                return None

            self._entries_today[symbol] = self._entries_today.get(symbol, 0) + 1
            self._total_positions += 1
            del self._detected_boxes[symbol]

            return IntradaySetup(
                symbol=symbol, strategy_id=self.strategy_id, direction='short',
                entry_price=round(entry_price, 2), stop_price=round(stop_price, 2),
                target_price=round(target_price, 2), risk_reward=round(actual_rr, 2),
                confidence=round(confidence, 2), setup_type='box_breakout_short',
                indicators={'box_high': round(box['high'], 2), 'box_low': round(box['low'], 2),
                            'box_height_pct': round(box['height_pct'] * 100, 2),
                            'box_bars': box['bars'], 'vol_surge': round(vol_surge, 2),
                            'vwap': round(vwap, 2), 'atr': round(atr, 4)},
                timestamp=now,
                notes=f'Box [{box["low"]:.2f}-{box["high"]:.2f}] {box["bars"]} bars, vol {vol_surge:.1f}x',
            )

        return None

    def _calc_confidence(self, box: Dict, vol_surge: float, vwap: float,
                          price: float, direction: str) -> float:
        """Calculate confidence score (0-1) based on multiple factors."""
        score = 0.3  # base

        # Volume surge quality
        if vol_surge >= 2.5:
            score += 0.2
        elif vol_surge >= 2.0:
            score += 0.15
        elif vol_surge >= 1.5:
            score += 0.1

        # Box tightness (tighter = better breakout potential)
        if box['height_pct'] <= 0.005:
            score += 0.15  # very tight box
        elif box['height_pct'] <= 0.008:
            score += 0.1

        # Number of bars (more consolidation = more energy)
        if box['bars'] >= 8:
            score += 0.15
        elif box['bars'] >= 6:
            score += 0.1

        # VWAP alignment
        if vwap > 0:
            if direction == 'long' and price > vwap:
                score += 0.1  # breaking out above VWAP = bullish
            elif direction == 'short' and price < vwap:
                score += 0.1  # breaking down below VWAP = bearish

        return min(0.95, score)

    def evaluate_exit(self, position: Dict, tick_data: Dict,
                      current_price: float, now: datetime) -> Optional[Tuple[bool, str]]:
        """Check for time exit."""
        cfg = self._config
        if now.hour >= cfg.get('eod_exit_hour', 15):
            if now.minute >= cfg.get('eod_exit_min', 50):
                self._total_positions = max(0, self._total_positions - 1)
                return (True, 'eod_exit')
        return None

    def validate_setup(self, setup: IntradaySetup, tick_data: Dict,
                       now: datetime) -> Tuple[bool, str]:
        if setup.risk_reward < 1.0:
            return (False, f'R:R {setup.risk_reward:.1f} too low')
        if setup.confidence < self._config.get('min_confidence', 0.4):
            return (False, f'Confidence {setup.confidence:.2f} too low')
        return (True, 'valid')

    def reset_daily(self):
        """Reset all daily state."""
        self._entries_today.clear()
        self._bar_history.clear()
        self._last_bar_time.clear()
        self._current_15m.clear()
        self._detected_boxes.clear()
        self._total_positions = 0
