"""Real-time indicator engine for gap fade strategies.

Computes VWAP, EMA, Opening Range, Day High/Low, RSI, ATR, and Volume Surge
from the Alpaca tick stream.
Only instantiated when a strategy requires indicators (classic strategy skips this).
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Market open time (ET)
MARKET_OPEN = time(9, 30)


@dataclass
class FiveMinBar:
    """Aggregated 5-minute OHLCV bar."""
    open: float = 0.0
    high: float = 0.0
    low: float = float('inf')
    close: float = 0.0
    volume: int = 0
    bar_time: str = ''  # 'HH:MM' of bar start


@dataclass
class SymbolIndicators:
    """Per-symbol indicator state."""
    # VWAP
    cum_price_volume: float = 0.0
    cum_volume: int = 0
    vwap: float = 0.0

    # Day high / low
    day_high: float = 0.0
    day_low: float = float('inf')

    # Opening Range (first N minutes)
    or_high: float = 0.0
    or_low: float = float('inf')
    or_complete: bool = False

    # 5-min bar aggregation
    current_bar: Optional[FiveMinBar] = field(default=None)
    completed_bars: List[FiveMinBar] = field(default_factory=list)

    # EMA state (single-period, backward compat)
    ema_value: float = 0.0
    ema_initialized: bool = False

    # Multi-period EMA state
    ema_values: Dict[int, float] = field(default_factory=dict)
    ema_initialized_flags: Dict[int, bool] = field(default_factory=dict)

    # RSI state (Wilder's smoothed)
    rsi: float = 50.0
    rsi_initialized: bool = False
    rsi_avg_gain: float = 0.0
    rsi_avg_loss: float = 0.0
    rsi_prev_close: float = 0.0
    rsi_bar_count: int = 0
    rsi_history: List[float] = field(default_factory=list)

    # ATR state
    atr: float = 0.0
    atr_initialized: bool = False
    atr_prev_close: float = 0.0
    atr_bar_count: int = 0

    # Volume surge
    volume_avg_20bar: float = 0.0


class TickIndicatorEngine:
    """Computes real-time indicators from trade ticks.

    Usage:
        engine = TickIndicatorEngine(
            indicators=['vwap', 'ema', 'opening_range', 'day_high', 'rsi', 'atr', 'volume_profile'],
            ema_period=9,
            ema_periods=[9, 20, 50, 200],
            rsi_period=14,
            atr_period=14,
            or_minutes=5,
        )
        # Feed ticks from AlpacaTickStreamer:
        engine.on_tick('AAPL', 150.25, size=100, timestamp=datetime(...))
        # Query:
        data = engine.get_data('AAPL')
        # data = {'vwap': 150.10, 'ema9': 149.80, 'ema20': ..., 'rsi': 55.3, 'atr': 2.1, ...}
    """

    def __init__(self, indicators: List[str], ema_period: int = 9,
                 or_minutes: int = 5, ema_buffer_pct: float = 0.001,
                 ema_periods: Optional[List[int]] = None,
                 rsi_period: int = 14, atr_period: int = 14):
        """
        Args:
            indicators: list of required indicators
            ema_period: EMA lookback for 5-min candles (default 9, backward compat)
            or_minutes: opening range window in minutes (default 5)
            ema_buffer_pct: buffer % added to EMA for trailing stop
            ema_periods: list of EMA periods for multi-period support (e.g. [9, 20, 50])
            rsi_period: RSI lookback period (default 14)
            atr_period: ATR lookback period (default 14)
        """
        self._indicators = set(indicators)
        self._ema_period = ema_period
        self._or_minutes = or_minutes
        self._ema_buffer_pct = ema_buffer_pct
        self._ema_multiplier = 2.0 / (ema_period + 1)
        self._rsi_period = rsi_period
        self._atr_period = atr_period

        # Multi-period EMA: always include the primary ema_period
        self._ema_periods: List[int] = []
        if 'ema' in self._indicators or 'ema_multi' in self._indicators:
            periods_set = {ema_period}
            if ema_periods:
                periods_set.update(ema_periods)
            self._ema_periods = sorted(periods_set)
        self._ema_multipliers = {p: 2.0 / (p + 1) for p in self._ema_periods}

        self._symbols: Dict[str, SymbolIndicators] = defaultdict(SymbolIndicators)
        self._date: str = ''  # current trading date (for daily reset)

    def reset(self):
        """Reset all indicator state (call on new trading day)."""
        self._symbols.clear()
        self._date = ''

    def reset_symbol(self, symbol: str):
        """Reset indicators for a single symbol."""
        if symbol in self._symbols:
            del self._symbols[symbol]

    def on_tick(self, symbol: str, price: float, size: int = 0,
                timestamp: Optional[datetime] = None):
        """Process a trade tick.

        Args:
            symbol: ticker symbol
            price: trade price
            size: trade size (shares) — needed for accurate VWAP
            timestamp: trade timestamp (ET). If None, uses current time.
        """
        if timestamp is None:
            from zoneinfo import ZoneInfo
            timestamp = datetime.now(ZoneInfo('US/Eastern'))

        # Daily reset check
        today = timestamp.strftime('%Y-%m-%d')
        if today != self._date:
            self.reset()
            self._date = today

        si = self._symbols[symbol]

        # ── VWAP ──
        if 'vwap' in self._indicators and size > 0:
            si.cum_price_volume += price * size
            si.cum_volume += size
            if si.cum_volume > 0:
                si.vwap = si.cum_price_volume / si.cum_volume

        # ── Day High ──
        if 'day_high' in self._indicators:
            if price > si.day_high:
                si.day_high = price

        # ── Day Low ──
        if 'day_low' in self._indicators:
            if price < si.day_low:
                si.day_low = price

        # ── Opening Range ──
        if 'opening_range' in self._indicators and not si.or_complete:
            t = timestamp.time()
            or_end = time(9, 30 + self._or_minutes)
            if t >= MARKET_OPEN:
                if t < or_end:
                    if price > si.or_high:
                        si.or_high = price
                    if price < si.or_low:
                        si.or_low = price
                else:
                    si.or_complete = True
                    if si.or_high > 0 and si.or_low < float('inf'):
                        logger.debug(f"{symbol} Opening Range: "
                                     f"${si.or_high:.2f} / ${si.or_low:.2f}")

        # ── 5-min Bar Aggregation + EMA/RSI/ATR ──
        needs_bars = (self._indicators & {'ema', 'ema_multi', 'bar_history',
                                           'rsi', 'atr', 'volume_profile'})
        if needs_bars:
            self._update_bars(symbol, price, size, timestamp, si)

    def _update_bars(self, symbol: str, price: float, size: int,
                     ts: datetime, si: SymbolIndicators):
        """Aggregate ticks into 5-min bars and update EMA/RSI/ATR."""
        # Determine which 5-min bar this tick belongs to
        bar_minute = (ts.minute // 5) * 5
        bar_key = f'{ts.hour}:{bar_minute:02d}'

        if si.current_bar is None or si.current_bar.bar_time != bar_key:
            # Complete previous bar
            if si.current_bar is not None and si.current_bar.volume > 0:
                completed = si.current_bar
                si.completed_bars.append(completed)
                self._update_ema(si, completed.close)
                self._update_multi_ema(si, completed.close)
                self._update_rsi(si, completed.close)
                self._update_atr(si, completed)
                self._update_volume_avg(si)

            # Start new bar
            si.current_bar = FiveMinBar(
                open=price, high=price, low=price,
                close=price, volume=size, bar_time=bar_key,
            )
        else:
            # Update current bar
            bar = si.current_bar
            if price > bar.high:
                bar.high = price
            if price < bar.low:
                bar.low = price
            bar.close = price
            bar.volume += size

    def _update_ema(self, si: SymbolIndicators, close: float):
        """Update single-period EMA with a completed bar's close price (backward compat)."""
        if not si.ema_initialized:
            # Use SMA of first N bars as seed
            if len(si.completed_bars) >= self._ema_period:
                sma = sum(b.close for b in si.completed_bars[-self._ema_period:]) / self._ema_period
                si.ema_value = sma
                si.ema_initialized = True
            else:
                # Not enough bars yet — use simple average of what we have
                si.ema_value = sum(b.close for b in si.completed_bars) / len(si.completed_bars)
        else:
            si.ema_value = (close * self._ema_multiplier +
                            si.ema_value * (1 - self._ema_multiplier))

    def _update_multi_ema(self, si: SymbolIndicators, close: float):
        """Update all multi-period EMAs."""
        n_bars = len(si.completed_bars)
        for period in self._ema_periods:
            mult = self._ema_multipliers[period]
            if not si.ema_initialized_flags.get(period, False):
                if n_bars >= period:
                    sma = sum(b.close for b in si.completed_bars[-period:]) / period
                    si.ema_values[period] = sma
                    si.ema_initialized_flags[period] = True
                elif n_bars > 0:
                    si.ema_values[period] = sum(b.close for b in si.completed_bars) / n_bars
            else:
                prev = si.ema_values.get(period, close)
                si.ema_values[period] = close * mult + prev * (1 - mult)

    def _update_rsi(self, si: SymbolIndicators, close: float):
        """Update RSI using Wilder's smoothed method on completed 5-min bar closes."""
        if 'rsi' not in self._indicators:
            return

        si.rsi_bar_count += 1

        if si.rsi_bar_count == 1:
            # First bar — just store price, no change yet
            si.rsi_prev_close = close
            return

        change = close - si.rsi_prev_close
        si.rsi_prev_close = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        period = self._rsi_period

        if not si.rsi_initialized:
            # Accumulate gains/losses for initial SMA
            si.rsi_avg_gain += gain
            si.rsi_avg_loss += loss

            if si.rsi_bar_count - 1 >= period:
                # We have enough changes (bar_count - 1 changes from bar_count bars)
                si.rsi_avg_gain /= period
                si.rsi_avg_loss /= period
                si.rsi_initialized = True
                if si.rsi_avg_loss == 0:
                    si.rsi = 100.0
                else:
                    rs = si.rsi_avg_gain / si.rsi_avg_loss
                    si.rsi = 100.0 - (100.0 / (1.0 + rs))
                si.rsi_history.append(si.rsi)
                if len(si.rsi_history) > 20:
                    si.rsi_history = si.rsi_history[-20:]
        else:
            # Wilder's smoothing
            si.rsi_avg_gain = (si.rsi_avg_gain * (period - 1) + gain) / period
            si.rsi_avg_loss = (si.rsi_avg_loss * (period - 1) + loss) / period
            if si.rsi_avg_loss == 0:
                si.rsi = 100.0
            else:
                rs = si.rsi_avg_gain / si.rsi_avg_loss
                si.rsi = 100.0 - (100.0 / (1.0 + rs))
            si.rsi_history.append(si.rsi)
            if len(si.rsi_history) > 20:
                si.rsi_history = si.rsi_history[-20:]

    def _update_atr(self, si: SymbolIndicators, bar: FiveMinBar):
        """Update ATR using Wilder's smoothed method on completed 5-min bars."""
        if 'atr' not in self._indicators:
            return

        si.atr_bar_count += 1

        if si.atr_bar_count == 1:
            # First bar — true range is just high - low
            si.atr_prev_close = bar.close
            si.atr = bar.high - bar.low
            return

        # True Range = max(H-L, |H-prev_close|, |L-prev_close|)
        tr = max(
            bar.high - bar.low,
            abs(bar.high - si.atr_prev_close),
            abs(bar.low - si.atr_prev_close),
        )
        si.atr_prev_close = bar.close

        period = self._atr_period

        if not si.atr_initialized:
            if si.atr_bar_count >= period:
                # Use SMA of first N TRs as seed — approximate by averaging ATR so far
                # ATR was accumulated as running average
                si.atr = ((si.atr * (si.atr_bar_count - 1)) + tr) / si.atr_bar_count
                si.atr_initialized = True
            else:
                # Running average until we have enough bars
                si.atr = ((si.atr * (si.atr_bar_count - 1)) + tr) / si.atr_bar_count
        else:
            # Wilder's smoothing: ATR = (prev_ATR * (N-1) + TR) / N
            si.atr = (si.atr * (period - 1) + tr) / period

    def _update_volume_avg(self, si: SymbolIndicators):
        """Update 20-bar rolling volume average."""
        if 'volume_profile' not in self._indicators:
            return

        bars = si.completed_bars
        window = bars[-20:] if len(bars) >= 20 else bars
        if window:
            si.volume_avg_20bar = sum(b.volume for b in window) / len(window)

    def seed_bars(self, symbol: str, bars: List[dict], today: str = ''):
        """Seed indicator state from historical 5-min bars (e.g. fetched from Alpaca).

        This brings the engine to a warm state immediately on startup, avoiding
        the ~75-minute cold-start wait for RSI initialization.

        Args:
            symbol: ticker symbol
            bars: list of dicts with keys {open, high, low, close, volume, timestamp}
                  ordered chronologically (oldest first)
            today: date string 'YYYY-MM-DD' — sets the engine's current date
        """
        if not bars:
            return

        if today and today != self._date:
            self._date = today

        si = self._symbols[symbol]

        for b in bars:
            bar = FiveMinBar(
                open=b['open'], high=b['high'], low=b['low'],
                close=b['close'], volume=int(b.get('volume', 0)),
                bar_time=b.get('bar_time', ''),
            )
            si.completed_bars.append(bar)

            # Update bar-level indicators
            self._update_ema(si, bar.close)
            self._update_multi_ema(si, bar.close)
            self._update_rsi(si, bar.close)
            self._update_atr(si, bar)
            self._update_volume_avg(si)

            # VWAP cumulative state
            if 'vwap' in self._indicators and bar.volume > 0:
                si.cum_price_volume += bar.close * bar.volume
                si.cum_volume += bar.volume
                si.vwap = si.cum_price_volume / si.cum_volume

            # Day high/low
            if 'day_high' in self._indicators and bar.high > si.day_high:
                si.day_high = bar.high
            if 'day_low' in self._indicators and bar.low < si.day_low:
                si.day_low = bar.low

            # Opening range from bars within first or_minutes of 09:30
            if 'opening_range' in self._indicators and not si.or_complete:
                bt = b.get('bar_time', '')
                if bt:
                    try:
                        h, m = int(bt.split(':')[0]), int(bt.split(':')[1])
                        bar_t = time(h, m)
                        or_end = time(9, 30 + self._or_minutes)
                        if bar_t >= MARKET_OPEN and bar_t < or_end:
                            if bar.high > si.or_high:
                                si.or_high = bar.high
                            if bar.low < si.or_low:
                                si.or_low = bar.low
                        elif bar_t >= or_end:
                            si.or_complete = True
                    except (ValueError, IndexError):
                        pass

        logger.info(f"Seeded {symbol}: {len(bars)} bars, "
                    f"RSI={'%.1f' % si.rsi if si.rsi_initialized else 'pending'}, "
                    f"bars={len(si.completed_bars)}")

    def get_data(self, symbol: str) -> Dict:
        """Get current indicator values for a symbol.

        Returns dict with available indicators:
            {
                'vwap': float,
                'ema9': float,
                'ema_initialized': bool,
                'or_high': float,
                'or_low': float,
                'or_complete': bool,
                'day_high': float,
                'bar_count': int,
                'ema20': float, 'ema50': float, 'ema200': float,  # if ema_multi
                'rsi': float,
                'atr': float,
                'volume_surge_ratio': float,
                'volume_avg_20bar': float,
            }
        """
        si = self._symbols.get(symbol)
        if si is None:
            return {}

        data = {}

        if 'vwap' in self._indicators:
            data['vwap'] = si.vwap
            data['cum_volume'] = si.cum_volume

        if 'ema' in self._indicators:
            data[f'ema{self._ema_period}'] = si.ema_value
            data['ema_initialized'] = si.ema_initialized
            data['bar_count'] = len(si.completed_bars)
            # Also provide EMA + buffer for trailing stop use
            if si.ema_value > 0:
                data['ema_stop_short'] = si.ema_value * (1 + self._ema_buffer_pct)
                data['ema_stop_long'] = si.ema_value * (1 - self._ema_buffer_pct)

        # Multi-period EMA values
        if 'ema_multi' in self._indicators or 'ema' in self._indicators:
            for period in self._ema_periods:
                val = si.ema_values.get(period, 0.0)
                data[f'ema{period}'] = val
                initialized = si.ema_initialized_flags.get(period, False)
                data[f'ema{period}_initialized'] = initialized

        if 'opening_range' in self._indicators:
            data['or_high'] = si.or_high if si.or_high > 0 else None
            data['or_low'] = si.or_low if si.or_low < float('inf') else None
            data['or_complete'] = si.or_complete

        if 'day_high' in self._indicators:
            data['day_high'] = si.day_high

        if 'day_low' in self._indicators:
            data['day_low'] = si.day_low if si.day_low < float('inf') else None

        if 'bar_history' in self._indicators:
            data['bar_history'] = list(si.completed_bars)
            data['bar_count'] = len(si.completed_bars)

        if 'rsi' in self._indicators:
            data['rsi'] = si.rsi
            data['rsi_initialized'] = si.rsi_initialized
            data['rsi_history'] = list(si.rsi_history)

        if 'atr' in self._indicators:
            data['atr'] = si.atr
            data['atr_initialized'] = si.atr_initialized

        if 'volume_profile' in self._indicators:
            data['volume_avg_20bar'] = si.volume_avg_20bar
            # Volume surge ratio: current bar volume / 20-bar avg
            current_vol = si.current_bar.volume if si.current_bar else 0
            if si.volume_avg_20bar > 0 and current_vol > 0:
                data['volume_surge_ratio'] = current_vol / si.volume_avg_20bar
            else:
                data['volume_surge_ratio'] = 0.0

        # Always include bar_count if bars are tracked
        if 'bar_count' not in data and si.completed_bars:
            data['bar_count'] = len(si.completed_bars)

        return data

    def get_all_data(self) -> Dict[str, Dict]:
        """Get indicator data for all tracked symbols."""
        return {sym: self.get_data(sym) for sym in self._symbols}

    @property
    def active_symbols(self) -> List[str]:
        """Symbols currently being tracked."""
        return list(self._symbols.keys())
