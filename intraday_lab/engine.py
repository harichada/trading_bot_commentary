"""Intraday Replay Engine — reads 1-min bars, feeds strategies, tracks trades.

This is the core simulation loop. It replays historical minute bars for a given
date range and symbol set, feeding each bar to the selected strategy's scan method.
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, time as dtime
from typing import Dict, List, Optional, Tuple

import psycopg2

logger = logging.getLogger('IntradayLab')


@dataclass
class LabBar:
    """Single 1-min OHLCV bar."""
    symbol: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class LabTrade:
    """Completed trade."""
    symbol: str
    strategy_id: str
    direction: str
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    shares: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    hold_minutes: int
    indicators: Dict = field(default_factory=dict)


@dataclass
class LabPosition:
    """Open position during replay."""
    symbol: str
    strategy_id: str
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    entry_time: datetime
    shares: int
    high_water: float = 0.0
    indicators: Dict = field(default_factory=dict)


class BarLoader:
    """Load minute bars from PostgreSQL."""

    def __init__(self, db_url: str):
        self.db_url = db_url
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
        return self._conn

    def load_day(self, symbol: str, date: str) -> List[LabBar]:
        """Load all 1-min bars for a symbol on a given date (RTH only: 9:30-16:00)."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol, ts, open, high, low, close, volume
            FROM minute_bars
            WHERE symbol = %s AND ts::date = %s
              AND ts::time >= '09:30' AND ts::time < '16:00'
            ORDER BY ts
        """, (symbol, date))
        bars = [LabBar(symbol=r[0], ts=r[1], open=r[2], high=r[3],
                        low=r[4], close=r[5], volume=r[6]) for r in cur.fetchall()]
        return bars

    def load_day_batch(self, symbols: List[str], date: str,
                       start_time: str = '09:30', end_time: str = '15:50') -> Dict[str, List['LabBar']]:
        """Load bars for ALL symbols on a single date in ONE query. Much faster."""
        conn = self._get_conn()
        cur = conn.cursor()
        if not symbols:
            return {}
        placeholders = ','.join(['%s'] * len(symbols))
        cur.execute(f"""
            SELECT symbol, ts, open, high, low, close, volume
            FROM minute_bars
            WHERE symbol IN ({placeholders}) AND ts::date = %s
              AND ts::time >= %s AND ts::time <= %s
            ORDER BY symbol, ts
        """, (*symbols, date, start_time, end_time))
        result: Dict[str, List[LabBar]] = {}
        for r in cur.fetchall():
            bar = LabBar(symbol=r[0], ts=r[1], open=r[2], high=r[3],
                         low=r[4], close=r[5], volume=r[6])
            result.setdefault(r[0], []).append(bar)
        return result

    def load_range(self, symbol: str, start: str, end: str,
                   start_time: str = '10:00', end_time: str = '15:50') -> List[LabBar]:
        """Load bars for a symbol across a date range, filtered to intraday window."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol, ts, open, high, low, close, volume
            FROM minute_bars
            WHERE symbol = %s AND ts::date >= %s AND ts::date <= %s
              AND ts::time >= %s AND ts::time <= %s
            ORDER BY ts
        """, (symbol, start, end, start_time, end_time))
        return [LabBar(symbol=r[0], ts=r[1], open=r[2], high=r[3],
                        low=r[4], close=r[5], volume=r[6]) for r in cur.fetchall()]

    _dates_cache: Dict[str, List[str]] = {}
    _symbols_cache: Dict[str, List[str]] = {}

    def get_trading_dates(self, start: str, end: str) -> List[str]:
        """Get trading dates — uses daily_bars table (fast) instead of minute_bars."""
        cache_key = f'{start}_{end}'
        if cache_key in self._dates_cache:
            return self._dates_cache[cache_key]
        conn = self._get_conn()
        cur = conn.cursor()
        # Use daily_bars table which is much smaller and indexed
        cur.execute("""
            SELECT DISTINCT date FROM daily_bars
            WHERE date >= %s AND date <= %s
            ORDER BY 1
        """, (start, end))
        dates = [str(r[0]) for r in cur.fetchall()]
        self._dates_cache[cache_key] = dates
        return dates

    # Curated intraday watchlist — high liquidity, volatile, tight spreads
    INTRADAY_CORE = [
        # Mega cap tech (highest intraday volume + volatility)
        'TSLA', 'NVDA', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOG', 'GOOGL',
        # High beta / volatile large caps
        'PLTR', 'AMD', 'DELL', 'MU', 'AVGO', 'CRM', 'NFLX', 'UBER',
        'COIN', 'SNOW', 'SHOP', 'SQ', 'ROKU', 'DKNG', 'MARA', 'RIOT',
        # Semiconductor / AI plays
        'ARM', 'SMCI', 'MRVL', 'QCOM', 'INTC', 'ON', 'ANET',
        # EV / Energy
        'RIVN', 'LCID', 'NIO', 'ENPH', 'FSLR',
        # Biotech movers
        'MRNA', 'BNTX',
        # Indices
        'SPY', 'QQQ', 'IWM',
        # Old school volatile
        'IBM', 'BA', 'GS', 'JPM',
    ]

    # Leveraged/inverse ETFs to exclude
    _LEVERAGED = {'TQQQ','SQQQ','SOXL','SOXS','SPXU','SPXS','TZA','TNA','UVXY','SVXY',
                  'LABU','LABD','FNGU','FNGD','SPDN','UPRO','UDOW','SDOW','QLD','QID',
                  'SSO','SDS','UCO','SCO','BOIL','KOLD','JNUG','JDST','DUST','NUGT',
                  'TSLQ','TSLL','NVDL','NVDD','PLTD','NVD','SVIX'}

    def get_symbols_for_date(self, date: str, min_bars: int = 100,
                              min_volume: int = 50000,
                              use_core: bool = True) -> List[str]:
        """Get symbols for intraday testing.

        Default: INTRADAY_CORE list (TSLA, NVDA, AAPL, etc.) + top daily movers.
        This gives consistent, liquid stocks that intraday strategies actually work on.
        """
        if date in self._symbols_cache:
            return self._symbols_cache[date]

        result = []

        # Start with core watchlist (always included if they traded that day)
        if use_core:
            conn = self._get_conn()
            cur = conn.cursor()
            placeholders = ','.join(['%s'] * len(self.INTRADAY_CORE))
            cur.execute(f"""
                SELECT symbol FROM daily_bars
                WHERE date = %s AND symbol IN ({placeholders}) AND volume > 0
            """, (date, *self.INTRADAY_CORE))
            result = [r[0] for r in cur.fetchall()]

        # Supplement with top movers (excluding leveraged + already included)
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT symbol FROM daily_bars
            WHERE date = %s AND volume >= %s AND close >= 5 AND close <= 500
            ORDER BY ABS(close - open) / NULLIF(open, 0) DESC
            LIMIT 100
        """, (date, min_volume))
        seen = set(result) | self._LEVERAGED
        for r in cur.fetchall():
            sym = r[0]
            if sym not in seen and not sym.endswith('W'):
                result.append(sym)
                seen.add(sym)
            if len(result) >= 200:
                break

        self._symbols_cache[date] = result
        return result

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()


class IndicatorState:
    """Lightweight indicator calculator for replay (replaces TickIndicatorEngine).

    Computes VWAP, EMA, RSI, volume profile from bars — no live stream needed.
    """

    def __init__(self):
        self.bars: List[LabBar] = []
        self.cum_vol = 0
        self.cum_vol_price = 0.0

    def update(self, bar: LabBar):
        self.bars.append(bar)
        self.cum_vol += bar.volume
        self.cum_vol_price += bar.close * bar.volume

    @property
    def vwap(self) -> float:
        return self.cum_vol_price / self.cum_vol if self.cum_vol > 0 else 0.0

    @property
    def last_price(self) -> float:
        return self.bars[-1].close if self.bars else 0.0

    @property
    def rsi(self) -> float:
        """14-bar RSI."""
        if len(self.bars) < 15:
            return 50.0
        closes = [b.close for b in self.bars[-15:]]
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            gains.append(d if d > 0 else 0)
            losses.append(-d if d < 0 else 0)
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @property
    def ema_9(self) -> float:
        return self._ema(9)

    @property
    def ema_21(self) -> float:
        return self._ema(21)

    def _sma(self, period: int) -> float:
        if len(self.bars) < period:
            return self.last_price
        return sum(b.close for b in self.bars[-period:]) / period

    def _stddev(self, period: int) -> float:
        if len(self.bars) < period:
            return 0.0
        closes = [b.close for b in self.bars[-period:]]
        mean = sum(closes) / len(closes)
        variance = sum((c - mean) ** 2 for c in closes) / len(closes)
        return variance ** 0.5

    def _rsi(self, period: int) -> float:
        if len(self.bars) < period + 1:
            return 50.0
        closes = [b.close for b in self.bars[-(period+1):]]
        gains, losses = [], []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            gains.append(d if d > 0 else 0)
            losses.append(-d if d < 0 else 0)
        avg_gain = sum(gains) / len(gains) if gains else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        if avg_loss == 0:
            return 100.0
        return 100 - (100 / (1 + avg_gain / avg_loss))

    def _ema(self, period: int) -> float:
        if len(self.bars) < period:
            return self.last_price
        closes = [b.close for b in self.bars[-period*2:]]
        k = 2 / (period + 1)
        ema = closes[0]
        for c in closes[1:]:
            ema = c * k + ema * (1 - k)
        return ema

    @property
    def volume_surge(self) -> float:
        """Current bar volume vs 20-bar average."""
        if len(self.bars) < 21:
            return 1.0
        avg = sum(b.volume for b in self.bars[-21:-1]) / 20
        return self.bars[-1].volume / avg if avg > 0 else 1.0

    @property
    def atr(self) -> float:
        """14-bar ATR."""
        if len(self.bars) < 15:
            return 0.0
        trs = []
        for i in range(-14, 0):
            h = self.bars[i].high
            l = self.bars[i].low
            pc = self.bars[i-1].close
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return sum(trs) / len(trs)

    def get_tick_data(self) -> Dict:
        """Return indicator snapshot in the format strategies expect.

        Provides BOTH naming conventions used by different strategies:
        - volume_surge AND volume_surge_ratio
        - ema_9/ema_21 AND ema9/ema21
        """
        vs = self.volume_surge
        return {
            'vwap': self.vwap,
            'rsi': self.rsi,
            'ema_9': self.ema_9,
            'ema_21': self.ema_21,
            'ema9': self.ema_9,
            'ema21': self.ema_21,
            'atr': self.atr,
            'volume_surge': vs,
            'volume_surge_ratio': vs,       # alias for strategies that use this name
            'vol_surge': vs,                 # another alias
            'price': self.last_price,
            'bar_count': len(self.bars),
            'cum_volume': self.cum_vol,
            'high_of_day': max((b.high for b in self.bars), default=0),
            'low_of_day': min((b.low for b in self.bars), default=999999),
            'open_price': self.bars[0].open if self.bars else 0,
            'or_high': max((b.high for b in self.bars[:30]), default=0) if len(self.bars) >= 30 else 0,
            'or_low': min((b.low for b in self.bars[:30]), default=999999) if len(self.bars) >= 30 else 0,
            'or_complete': len(self.bars) >= 30,  # 30 bars = 30 min opening range
            'trend': 'up' if self.ema_9 > self.ema_21 else 'down',
            'trend_direction': 'up' if self.ema_9 > self.ema_21 else 'down',
            'ema20': self._ema(20),
            'ema50': self._ema(50),
            'sma20': self._sma(20),
            'adx': 25.0,  # placeholder — full ADX needs directional movement
            'bb_upper': self._sma(20) + 2 * self._stddev(20),
            'bb_lower': self._sma(20) - 2 * self._stddev(20),
            'bb_mid': self._sma(20),
            'rsi2': self._rsi(2),
            'rsi14': self.rsi,
            'rsi_initialized': len(self.bars) >= 15,
            'daily_rsi': self.rsi,  # alias
            'day_high': max((b.high for b in self.bars), default=0),
            'day_low': min((b.low for b in self.bars), default=float('inf')),
            'prev_close': self.bars[0].open if self.bars else 0,
            'gap_pct': 0,  # filled by run_day if available
        }

    def reset(self):
        """Reset for a new day."""
        self.bars.clear()
        self.cum_vol = 0
        self.cum_vol_price = 0.0


class ReplayEngine:
    """Core replay engine — feeds bars to strategy, manages positions, tracks P&L."""

    def __init__(self, db_url: str, capital: float = 100000, risk_pct: float = 0.01,
                 max_positions: int = 3):
        self.loader = BarLoader(db_url)
        self.capital = capital
        self.equity = capital
        self.risk_pct = risk_pct
        self.max_positions = max_positions
        self.positions: Dict[str, LabPosition] = {}
        self.trades: List[LabTrade] = []
        self.indicators: Dict[str, IndicatorState] = {}

    def run_day(self, strategy, symbols: List[str], date: str,
                start_time: str = '10:00', end_time: str = '15:50') -> List[LabTrade]:
        """Replay one day for one strategy across multiple symbols."""
        day_trades = []

        # Batch load all symbols in ONE query (10-50x faster than per-symbol)
        all_bars = self.loader.load_day_batch(symbols, date, '09:30', end_time)

        if not all_bars:
            return day_trades

        # Reset indicators
        self.indicators = {sym: IndicatorState() for sym in all_bars}

        # Build unified timeline
        timeline: List[Tuple[datetime, str, LabBar]] = []
        for sym, bars in all_bars.items():
            for bar in bars:
                timeline.append((bar.ts, sym, bar))
        timeline.sort(key=lambda x: x[0])

        # Replay bar by bar
        start_dt = datetime.strptime(f'{date} {start_time}', '%Y-%m-%d %H:%M')

        for ts, sym, bar in timeline:
            # Update indicators (always, even before entry window)
            self.indicators[sym].update(bar)

            # Check exits on open positions
            if sym in self.positions:
                trade = self._check_exit(sym, bar, strategy)
                if trade:
                    day_trades.append(trade)

            # Only scan for entries after start_time
            if ts < start_dt:
                continue

            # Scan for setups
            if sym not in self.positions and len(self.positions) < self.max_positions:
                tick_data = self.indicators[sym].get_tick_data()
                try:
                    setup = strategy.scan_for_setups(
                        sym, tick_data,
                        {'price': bar.close, 'volume': bar.volume,
                         'open': bar.open, 'high': bar.high, 'low': bar.low},
                        bar.ts)
                    if setup:
                        valid, reason = strategy.validate_setup(setup, tick_data, bar.ts)
                        if valid:
                            self._enter(setup, bar)
                except Exception as e:
                    logger.debug(f"Strategy scan error {sym}: {e}")

        # Force close all remaining positions at end of day
        for sym in list(self.positions.keys()):
            last_bar = all_bars.get(sym, [None])[-1]
            if last_bar:
                trade = self._force_exit(sym, last_bar.close, last_bar.ts, 'eod')
                if trade:
                    day_trades.append(trade)

        return day_trades

    def _enter(self, setup, bar: LabBar):
        """Open a position from a setup signal."""
        risk_per_share = abs(setup.entry_price - setup.stop_price)
        if risk_per_share <= 0:
            return
        risk_amount = self.equity * self.risk_pct
        shares = max(1, int(risk_amount / risk_per_share))
        # Cap at 20% of equity per position
        max_shares = int(self.equity * 0.20 / setup.entry_price) if setup.entry_price > 0 else 1
        shares = min(shares, max_shares)

        self.positions[setup.symbol] = LabPosition(
            symbol=setup.symbol,
            strategy_id=setup.strategy_id,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            entry_time=bar.ts,
            shares=shares,
            high_water=setup.entry_price,
            indicators=setup.indicators,
        )

    def _check_exit(self, sym: str, bar: LabBar, strategy) -> Optional[LabTrade]:
        """Check stop, target, and strategy-specific exits."""
        pos = self.positions.get(sym)
        if not pos:
            return None

        price = bar.close

        # Update high water mark
        if pos.direction == 'long':
            pos.high_water = max(pos.high_water, bar.high)
        else:
            pos.high_water = min(pos.high_water, bar.low) if pos.high_water > 0 else bar.low

        # Stop loss
        if pos.direction == 'long' and bar.low <= pos.stop_price:
            return self._force_exit(sym, pos.stop_price, bar.ts, 'stop_loss')
        if pos.direction == 'short' and bar.high >= pos.stop_price:
            return self._force_exit(sym, pos.stop_price, bar.ts, 'stop_loss')

        # Target
        if pos.direction == 'long' and bar.high >= pos.target_price:
            return self._force_exit(sym, pos.target_price, bar.ts, 'target')
        if pos.direction == 'short' and bar.low <= pos.target_price:
            return self._force_exit(sym, pos.target_price, bar.ts, 'target')

        # Strategy-specific exit
        try:
            tick_data = self.indicators[sym].get_tick_data()
            should_exit = strategy.evaluate_exit(
                asdict(pos) if hasattr(pos, '__dataclass_fields__') else {},
                tick_data, price, bar.ts)
            if should_exit and isinstance(should_exit, tuple) and should_exit[0]:
                reason = should_exit[1] if len(should_exit) > 1 else 'strategy_exit'
                return self._force_exit(sym, price, bar.ts, reason)
        except Exception:
            pass  # Strategy doesn't implement evaluate_exit or it errors

        return None

    def _force_exit(self, sym: str, exit_price: float, ts: datetime,
                    reason: str) -> Optional[LabTrade]:
        """Close a position and record the trade."""
        pos = self.positions.pop(sym, None)
        if not pos:
            return None

        if pos.direction == 'long':
            pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            pnl = (pos.entry_price - exit_price) * pos.shares

        pnl_pct = pnl / (pos.entry_price * pos.shares) if pos.entry_price * pos.shares > 0 else 0
        self.equity += pnl
        hold = int((ts - pos.entry_time).total_seconds() / 60) if isinstance(ts, datetime) else 0

        trade = LabTrade(
            symbol=sym, strategy_id=pos.strategy_id, direction=pos.direction,
            entry_price=pos.entry_price, exit_price=exit_price,
            entry_time=pos.entry_time.strftime('%Y-%m-%d %H:%M'),
            exit_time=ts.strftime('%Y-%m-%d %H:%M') if isinstance(ts, datetime) else str(ts),
            shares=pos.shares, pnl=round(pnl, 2), pnl_pct=round(pnl_pct, 6),
            exit_reason=reason, hold_minutes=hold, indicators=pos.indicators,
        )
        self.trades.append(trade)
        return trade

    def get_metrics(self) -> Dict:
        """Compute performance metrics from all trades."""
        if not self.trades:
            return {'total_trades': 0, 'net_pnl': 0, 'win_rate': 0, 'profit_factor': 0,
                    'sharpe': 0, 'max_drawdown_pct': 0, 'avg_win': 0, 'avg_loss': 0}

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        net = gross_profit - gross_loss

        # Drawdown
        peak = self.capital
        max_dd = 0
        running = self.capital
        daily_returns = []
        for t in self.trades:
            running += t.pnl
            peak = max(peak, running)
            dd = (peak - running) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
            daily_returns.append(t.pnl / self.capital)

        # Sharpe (annualized, 252 trading days)
        import statistics
        sharpe = 0
        if len(daily_returns) > 1:
            mean_r = statistics.mean(daily_returns)
            std_r = statistics.stdev(daily_returns)
            if std_r > 0:
                sharpe = (mean_r / std_r) * (252 ** 0.5)

        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(self.trades),
            'net_pnl': round(net, 2),
            'gross_profit': round(gross_profit, 2),
            'gross_loss': round(gross_loss, 2),
            'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999,
            'sharpe': round(sharpe, 2),
            'max_drawdown_pct': round(max_dd * 100, 1),
            'avg_win': round(gross_profit / len(wins), 2) if wins else 0,
            'avg_loss': round(-gross_loss / len(losses), 2) if losses else 0,
            'avg_hold_min': round(sum(t.hold_minutes for t in self.trades) / len(self.trades)),
            'final_equity': round(self.equity, 2),
            'return_pct': round((self.equity - self.capital) / self.capital * 100, 1),
        }

    def reset(self):
        """Reset for a new run."""
        self.equity = self.capital
        self.positions.clear()
        self.trades.clear()
        self.indicators.clear()
