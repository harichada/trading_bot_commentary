#!/usr/bin/env python3
"""
Rudra Swing Engine — Professional swing trading modules for the Rudra Trading Engine.

5 Priority-1 modules based on Weinstein Stage Analysis, EMA hierarchy, and volume confirmation.
Fundamentally different from gap fade/intraday: 3-6 trades/month, 1-4 week holds.

Modules:
    1. WeeklyKillSwitch  — System-wide halt when weekly bias breaks
    2. StageClassifier   — Weinstein 4-stage classification
    3. TightBaseDetector — Tight base / consolidation detection
    4. VolumeEngine      — Complete volume intelligence
    5. MultiTimeframeBias — Weekly bias check before daily entry

Usage:
    python rudra_swing_engine.py                          # Full backtest 2021-2026
    python rudra_swing_engine.py --scan                   # Current Stage 2 stocks
    python rudra_swing_engine.py --classify AAPL          # Classify one stock
    python rudra_swing_engine.py --start 2021-01-01       # Custom start date
    python rudra_swing_engine.py --capital 25000           # Custom capital
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rudra_swing")

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def _sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=window, min_periods=window).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def aggregate_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily bars to weekly OHLCV.

    Expects columns: date, open, high, low, close, volume.
    Returns DataFrame with columns: week_start, open, high, low, close, volume.
    """
    df = daily.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # ISO week grouping — use Monday as week start
    df["week_start"] = df["date"].dt.to_period("W-FRI").apply(lambda p: p.start_time)
    weekly = df.groupby("week_start").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()
    weekly = weekly.sort_values("week_start").reset_index(drop=True)
    return weekly


# ---------------------------------------------------------------------------
# Module 1: WeeklyKillSwitch
# ---------------------------------------------------------------------------

class WeeklyKillSwitch:
    """System-wide halt when weekly bias breaks.

    Rule: If SPY weekly close < weekly 21 EMA -> halt ALL new entries.
    When weekly bias is bearish, no strategy works reliably.
    Once broken, require a weekly close ABOVE the 21 EMA to re-enable.
    """

    EMA_PERIOD = 21

    def __init__(self) -> None:
        self._was_halted = False

    def is_active(self, spy_weekly_bars: pd.DataFrame) -> Tuple[bool, str]:
        """Returns (trading_allowed, reason).

        Parameters
        ----------
        spy_weekly_bars : DataFrame with columns [week_start, open, high, low, close, volume]
            Must have at least 21 rows.
        """
        if len(spy_weekly_bars) < self.EMA_PERIOD:
            return False, f"Insufficient SPY weekly data ({len(spy_weekly_bars)} bars, need {self.EMA_PERIOD})"

        closes = spy_weekly_bars["close"].astype(float)
        ema21 = _ema(closes, self.EMA_PERIOD)

        last_close = closes.iloc[-1]
        last_ema = ema21.iloc[-1]

        if last_close > last_ema:
            self._was_halted = False
            pct_above = (last_close / last_ema - 1) * 100
            return True, f"SPY weekly close {last_close:.2f} ABOVE 21 EMA {last_ema:.2f} (+{pct_above:.1f}%)"

        self._was_halted = True
        pct_below = (1 - last_close / last_ema) * 100
        return False, f"KILL SWITCH: SPY weekly close {last_close:.2f} BELOW 21 EMA {last_ema:.2f} (-{pct_below:.1f}%)"


# ---------------------------------------------------------------------------
# Module 2: StageClassifier (Weinstein 4-Stage)
# ---------------------------------------------------------------------------

class StageClassifier:
    """Classify every stock into Weinstein stages using weekly bars.

    Stage 1 (Accumulation): Sideways, flat 30-week MA, range-bound
    Stage 2 (Advancing): Price above rising 30-week MA, higher highs — ONLY TRADEABLE
    Stage 3 (Distribution): Price above but flattening 30-week MA, choppy
    Stage 4 (Decline): Price below declining 30-week MA — AVOID
    """

    MA_PERIOD = 30
    SLOPE_LOOKBACK = 4
    SIDEWAYS_THRESHOLD = 0.10  # 10% range = sideways
    SLOPE_FLAT_THRESHOLD = 0.005  # <0.5% slope over 4 weeks = flat

    def classify(self, symbol: str, weekly_bars: pd.DataFrame) -> str:
        """Returns 'STAGE_1', 'STAGE_2', 'STAGE_3', 'STAGE_4', or 'UNCLEAR'."""
        if len(weekly_bars) < self.MA_PERIOD + self.SLOPE_LOOKBACK:
            return "UNCLEAR"

        closes = weekly_bars["close"].astype(float)
        highs = weekly_bars["high"].astype(float)

        ma30 = _sma(closes, self.MA_PERIOD)
        last_close = closes.iloc[-1]
        last_ma = ma30.iloc[-1]
        prev_ma = ma30.iloc[-1 - self.SLOPE_LOOKBACK]

        if pd.isna(last_ma) or pd.isna(prev_ma) or prev_ma == 0:
            return "UNCLEAR"

        slope = (last_ma - prev_ma) / prev_ma
        price_above_ma = last_close > last_ma

        # Higher highs check — last 3 weekly highs increasing
        recent_highs = highs.iloc[-3:].values
        higher_highs = all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs)))

        # Sideways check — 10-week price range < threshold
        recent_10 = closes.iloc[-10:]
        price_range = (recent_10.max() - recent_10.min()) / recent_10.mean() if recent_10.mean() > 0 else 0

        # Lower lows check
        recent_lows = weekly_bars["low"].astype(float).iloc[-3:].values
        lower_lows = all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows)))

        # Classification logic
        if price_above_ma and slope > self.SLOPE_FLAT_THRESHOLD and higher_highs:
            return "STAGE_2"

        if not price_above_ma and slope < -self.SLOPE_FLAT_THRESHOLD:
            return "STAGE_4"

        if price_above_ma and abs(slope) <= self.SLOPE_FLAT_THRESHOLD:
            # Flattening MA, price still above — distribution
            return "STAGE_3"

        if abs(slope) <= self.SLOPE_FLAT_THRESHOLD and price_range < self.SIDEWAYS_THRESHOLD:
            return "STAGE_1"

        if price_above_ma and slope > self.SLOPE_FLAT_THRESHOLD:
            # Above rising MA but not making higher highs — early Stage 2 or late Stage 1
            return "STAGE_2"

        if not price_above_ma and abs(slope) <= self.SLOPE_FLAT_THRESHOLD:
            return "STAGE_1"

        return "UNCLEAR"

    def scan_universe(self, symbol_weekly_map: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Classify entire universe. Returns {symbol: stage}."""
        results = {}
        for symbol, weekly_bars in symbol_weekly_map.items():
            results[symbol] = self.classify(symbol, weekly_bars)
        return results


# ---------------------------------------------------------------------------
# Module 3: TightBaseDetector
# ---------------------------------------------------------------------------

class TightBaseDetector:
    """Detect tight base / consolidation patterns.

    A tight base forms when:
    1. Strong prior move exists (8%+ in last 10 days)
    2. Pullback occurs with declining volume
    3. Price compresses into tight range (ATR contracts)
    4. Volume dries up (< 60% of 20-day average)
    5. Price holds above key support levels
    """

    PRIOR_MOVE_MIN = 0.08       # 8% prior impulse
    PRIOR_MOVE_LOOKBACK = 20    # look back 20 bars for the impulse
    ATR_COMPRESS_RATIO = 0.60   # ATR(5) < ATR(20) * 0.60
    VOLUME_DRY_RATIO = 0.60     # 5-day avg vol < 20-day avg vol * 0.60
    BASE_RANGE_MAX = 0.03       # base range < 3% of close
    BASE_LOOKBACK_MIN = 3
    BASE_LOOKBACK_MAX = 10

    def detect(self, symbol: str, daily_bars: pd.DataFrame) -> Optional[dict]:
        """Returns setup dict if tight base detected, None otherwise."""
        if len(daily_bars) < 40:
            return None

        closes = daily_bars["close"].astype(float)
        highs = daily_bars["high"].astype(float)
        lows = daily_bars["low"].astype(float)
        volumes = daily_bars["volume"].astype(float)

        last_close = closes.iloc[-1]
        if last_close <= 0:
            return None

        # Prior strong move — max close in last 10 bars vs min close in 10 bars before that
        recent_max = closes.iloc[-20:-5].max() if len(closes) >= 20 else closes.iloc[:-5].max()
        prior_min = closes.iloc[-30:-15].min() if len(closes) >= 30 else closes.iloc[:max(1, len(closes) - 20)].min()
        if prior_min <= 0:
            return None
        prior_move_pct = (recent_max - prior_min) / prior_min
        if prior_move_pct < self.PRIOR_MOVE_MIN:
            return None

        # ATR compression
        atr5 = _atr(highs, lows, closes, 5)
        atr20 = _atr(highs, lows, closes, 20)
        last_atr5 = atr5.iloc[-1]
        last_atr20 = atr20.iloc[-1]
        if pd.isna(last_atr5) or pd.isna(last_atr20) or last_atr20 == 0:
            return None
        compression_ratio = last_atr5 / last_atr20

        # Volume dry-up
        vol_5 = volumes.iloc[-5:].mean()
        vol_20 = volumes.iloc[-20:].mean()
        if vol_20 == 0:
            return None
        volume_ratio = vol_5 / vol_20

        # Base range — look at last 5-10 bars
        base_high = highs.iloc[-self.BASE_LOOKBACK_MAX:].max()
        base_low = lows.iloc[-self.BASE_LOOKBACK_MAX:].min()
        base_range_pct = (base_high - base_low) / last_close if last_close > 0 else 999

        # Days in base — count bars where range < median daily range
        median_range = (highs - lows).iloc[-20:].median()
        days_in_base = 0
        for i in range(1, min(self.BASE_LOOKBACK_MAX + 1, len(daily_bars))):
            idx = -i
            bar_range = highs.iloc[idx] - lows.iloc[idx]
            if bar_range <= median_range * 1.2:
                days_in_base += 1
            else:
                break

        # Scoring — all factors contribute
        compress_score = max(0, (self.ATR_COMPRESS_RATIO - compression_ratio) / self.ATR_COMPRESS_RATIO) * 30
        vol_dry_score = max(0, (self.VOLUME_DRY_RATIO - volume_ratio) / self.VOLUME_DRY_RATIO) * 30
        range_score = max(0, (self.BASE_RANGE_MAX - base_range_pct) / self.BASE_RANGE_MAX) * 20
        move_score = min(prior_move_pct / 0.20, 1.0) * 20  # cap at 20% move
        score = compress_score + vol_dry_score + range_score + move_score

        # Require at least some compression and volume dry-up
        if compression_ratio > self.ATR_COMPRESS_RATIO * 1.3 and volume_ratio > self.VOLUME_DRY_RATIO * 1.3:
            return None

        if score < 15:
            return None

        return {
            "symbol": symbol,
            "base_high": float(base_high),
            "base_low": float(base_low),
            "compression_ratio": float(compression_ratio),
            "volume_ratio": float(volume_ratio),
            "prior_move_pct": float(prior_move_pct),
            "days_in_base": days_in_base,
            "score": float(score),
        }


# ---------------------------------------------------------------------------
# Module 4: VolumeEngine
# ---------------------------------------------------------------------------

class VolumeEngine:
    """Complete volume intelligence — no guessing.

    Volume decision table:
    - High volume + price up   = BIG MONEY BUYING (enter/hold)
    - High volume + price down = BIG MONEY SELLING (exit immediately)
    - Low volume + price up    = TRAP MOVE (skip/avoid)
    - Very low volume          = NO TRADE (wait)
    """

    HIGH_VOL_THRESHOLD = 1.50   # > 1.5x 20-day average
    LOW_VOL_THRESHOLD = 0.60    # < 0.6x 20-day average
    PULLBACK_VOL_THRESHOLD = 0.80

    def _vol_stats(self, bars: pd.DataFrame) -> Tuple[float, float, float]:
        """Returns (current_vol_ratio, price_change_pct, avg_vol_20)."""
        volumes = bars["volume"].astype(float)
        closes = bars["close"].astype(float)
        if len(bars) < 20 or volumes.iloc[-20:].mean() == 0:
            return 0.0, 0.0, 0.0
        avg20 = volumes.iloc[-20:].mean()
        curr_vol = volumes.iloc[-1]
        vol_ratio = curr_vol / avg20
        price_chg = (closes.iloc[-1] / closes.iloc[-2] - 1) if closes.iloc[-2] != 0 else 0
        return vol_ratio, price_chg, avg20

    def classify(self, symbol: str, bars: pd.DataFrame) -> str:
        """Returns volume state string."""
        if len(bars) < 21:
            return "NO_TRADE"

        vol_ratio, price_chg, _ = self._vol_stats(bars)

        if vol_ratio < self.LOW_VOL_THRESHOLD:
            return "NO_TRADE"

        if vol_ratio >= self.HIGH_VOL_THRESHOLD:
            if price_chg > 0.005:
                return "BREAKOUT_CONFIRMED"
            elif price_chg < -0.005:
                return "DISTRIBUTION"
            else:
                return "ACCUMULATION"

        if vol_ratio < self.PULLBACK_VOL_THRESHOLD and price_chg < -0.002:
            return "HEALTHY_PULLBACK"

        if vol_ratio < self.PULLBACK_VOL_THRESHOLD and price_chg > 0.005:
            return "TRAP_MOVE"

        return "ACCUMULATION"

    def is_breakout_valid(self, symbol: str, bars: pd.DataFrame) -> bool:
        """Is the current breakout backed by volume?"""
        vol_ratio, price_chg, _ = self._vol_stats(bars)
        return vol_ratio >= self.HIGH_VOL_THRESHOLD and price_chg > 0

    def is_pullback_healthy(self, symbol: str, bars: pd.DataFrame) -> bool:
        """Is the pullback on declining volume (bullish)?"""
        if len(bars) < 21:
            return False
        volumes = bars["volume"].astype(float)
        closes = bars["close"].astype(float)
        avg20 = volumes.iloc[-20:].mean()
        if avg20 == 0:
            return False
        # Last 3 bars declining
        recent_closes = closes.iloc[-3:].values
        is_pullback = recent_closes[-1] < recent_closes[0]
        avg_recent_vol = volumes.iloc[-3:].mean()
        low_vol = avg_recent_vol / avg20 < self.PULLBACK_VOL_THRESHOLD
        return is_pullback and low_vol

    def trend_health(self, symbol: str, bars: pd.DataFrame) -> str:
        """Is volume supporting the trend?"""
        if len(bars) < 25:
            return "INSUFFICIENT_DATA"

        closes = bars["close"].astype(float)
        volumes = bars["volume"].astype(float)

        # Compare last 5 bars to prior 5 bars
        price_rising = closes.iloc[-5:].mean() > closes.iloc[-10:-5].mean()
        vol_rising = volumes.iloc[-5:].mean() > volumes.iloc[-10:-5].mean()

        if price_rising and vol_rising:
            return "POWER_TREND"
        if price_rising and not vol_rising:
            return "WEAKNESS_APPEARING"
        if not price_rising and vol_rising:
            return "SELLING_PRESSURE"
        return "DEAD_MONEY"


# ---------------------------------------------------------------------------
# Module 5: MultiTimeframeBias
# ---------------------------------------------------------------------------

class MultiTimeframeBias:
    """Weekly bias check before any daily entry.

    Rule hierarchy:
    1. Weekly close > Weekly 21 EMA (REQUIRED — kill switch)
    2. Weekly close > Weekly 30 MA (Stage 2 confirmation)
    3. Daily close > 200 EMA (institutional baseline)
    4. Daily EMA(9) > EMA(21) > EMA(50) (trend alignment)

    All 4 must be true for a LONG entry.
    """

    def __init__(self, kill_switch: WeeklyKillSwitch) -> None:
        self._kill_switch = kill_switch

    def check_bias(
        self,
        symbol: str,
        daily_bars: pd.DataFrame,
        spy_weekly: pd.DataFrame,
        symbol_weekly: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Check all bias conditions for the symbol."""
        # 1. Kill switch
        system_active, ks_reason = self._kill_switch.is_active(spy_weekly)

        # 2. Stage 2 — symbol's weekly close > 30-week MA
        stage_2 = False
        if symbol_weekly is not None and len(symbol_weekly) >= 30:
            ma30 = _sma(symbol_weekly["close"].astype(float), 30)
            stage_2 = float(symbol_weekly["close"].iloc[-1]) > float(ma30.iloc[-1])

        # 3. Daily close > 200 EMA
        above_200 = False
        if len(daily_bars) >= 200:
            ema200 = _ema(daily_bars["close"].astype(float), 200)
            above_200 = float(daily_bars["close"].iloc[-1]) > float(ema200.iloc[-1])

        # 4. EMA alignment: 9 > 21 > 50
        ema_aligned = False
        if len(daily_bars) >= 50:
            closes = daily_bars["close"].astype(float)
            e9 = _ema(closes, 9).iloc[-1]
            e21 = _ema(closes, 21).iloc[-1]
            e50 = _ema(closes, 50).iloc[-1]
            ema_aligned = e9 > e21 > e50

        all_pass = system_active and stage_2 and above_200 and ema_aligned

        # Risk sizing: normal 1%, reduced 0.5% if any condition weak
        conditions_met = sum([system_active, stage_2, above_200, ema_aligned])
        risk_pct = 0.02 if conditions_met >= 4 else 0.01

        details_parts = []
        if not system_active:
            details_parts.append(f"KILL_SWITCH_OFF: {ks_reason}")
        if not stage_2:
            details_parts.append("NOT_STAGE_2")
        if not above_200:
            details_parts.append("BELOW_200_EMA")
        if not ema_aligned:
            details_parts.append("EMA_NOT_ALIGNED")

        return {
            "system_active": system_active,
            "stage_2": stage_2,
            "above_200": above_200,
            "ema_aligned": ema_aligned,
            "all_pass": all_pass,
            "risk_pct": risk_pct,
            "details": "; ".join(details_parts) if details_parts else "ALL_CLEAR",
        }


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

class DataLoader:
    """Load daily bars from PostgreSQL and aggregate to weekly."""

    def __init__(self, db_url: str = DB_URL) -> None:
        self._db_url = db_url

    def _connect(self):
        return psycopg2.connect(self._db_url)

    def load_daily(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: str = "2026-12-31",
        min_volume: float = 100_000,
        min_price: float = 5.0,
    ) -> pd.DataFrame:
        """Load daily bars. If symbols is None, loads liquid universe."""
        conn = self._connect()
        try:
            if symbols:
                placeholders = ",".join(["%s"] * len(symbols))
                query = f"""
                    SELECT symbol, date, open, high, low, close, volume
                    FROM daily_bars
                    WHERE symbol IN ({placeholders})
                      AND date >= %s AND date <= %s
                    ORDER BY symbol, date
                """
                params = list(symbols) + [start_date, end_date]
            else:
                # Load only liquid stocks
                query = """
                    SELECT symbol, date, open, high, low, close, volume
                    FROM daily_bars
                    WHERE date >= %s AND date <= %s
                      AND volume >= %s
                      AND close >= %s
                    ORDER BY symbol, date
                """
                params = [start_date, end_date, min_volume, min_price]
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def load_spy_daily(self, start_date: str = "2020-01-01", end_date: str = "2026-12-31") -> pd.DataFrame:
        """Load SPY daily bars."""
        return self.load_daily(symbols=["SPY"], start_date=start_date, end_date=end_date)

    def get_liquid_symbols(
        self,
        as_of_date: str,
        lookback_days: int = 60,
        min_avg_volume: float = 500_000,
        min_price: float = 10.0,
        max_symbols: int = 2000,
    ) -> List[str]:
        """Get symbols meeting liquidity thresholds as of a date."""
        conn = self._connect()
        try:
            start = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            query = """
                SELECT symbol, AVG(volume) as avg_vol, MAX(close) as last_close
                FROM daily_bars
                WHERE date >= %s AND date <= %s
                  AND volume > 0 AND close > 0
                GROUP BY symbol
                HAVING AVG(volume) >= %s AND MAX(close) >= %s
                ORDER BY AVG(volume) DESC
                LIMIT %s
            """
            cur = conn.cursor()
            cur.execute(query, (start, as_of_date, min_avg_volume, min_price, max_symbols))
            rows = cur.fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Position / Trade Tracking
# ---------------------------------------------------------------------------

@dataclass
class SwingPosition:
    """An open swing position."""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    stop_price: float
    target_price: float
    initial_risk: float        # entry - stop per share
    partial_exited: bool = False
    stop_at_breakeven: bool = False

    @property
    def risk_amount(self) -> float:
        return self.initial_risk * self.shares


@dataclass
class SwingTrade:
    """A completed swing trade."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_pct: float
    r_multiple: float
    exit_reason: str
    holding_days: int


# ---------------------------------------------------------------------------
# Backtester
# ---------------------------------------------------------------------------

class SwingBacktester:
    """Backtest the swing strategy using all 5 modules."""

    MAX_POSITIONS = 8
    MAX_TRADES_PER_WEEK = 5
    RISK_PER_TRADE = 0.02   # 2% of equity
    PARTIAL_EXIT_R = 2.0
    BREAKEVEN_R = 1.0

    def __init__(
        self,
        start_date: str = "2021-01-01",
        end_date: str = "2026-03-21",
        initial_capital: float = 25_000.0,
        db_url: str = DB_URL,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.db_url = db_url

        self.kill_switch = WeeklyKillSwitch()
        self.classifier = StageClassifier()
        self.base_detector = TightBaseDetector()
        self.volume_engine = VolumeEngine()
        self.bias = MultiTimeframeBias(self.kill_switch)
        self.loader = DataLoader(db_url)

        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.positions: List[SwingPosition] = []
        self.trades: List[SwingTrade] = []
        self.equity_curve: List[Tuple[str, float]] = []
        self.weekly_stage2_counts: List[Tuple[str, int]] = []

    def run(self) -> None:
        """Execute the full backtest."""
        logger.info(
            "Starting swing backtest: %s to %s, capital=$%.0f",
            self.start_date, self.end_date, self.initial_capital,
        )

        # Need extra lookback for 30-week MA + 200-day EMA
        lookback_start = (
            datetime.strptime(self.start_date, "%Y-%m-%d") - timedelta(days=400)
        ).strftime("%Y-%m-%d")

        # Load SPY for kill switch
        logger.info("Loading SPY data...")
        spy_daily = self.loader.load_daily(
            symbols=["SPY"], start_date=lookback_start, end_date=self.end_date,
        )
        spy_weekly = aggregate_to_weekly(spy_daily)

        # Get liquid universe — refresh quarterly for survivorship-free coverage
        logger.info("Identifying liquid universe (refreshing quarterly)...")
        liquid_symbols_set: set = set()
        refresh_date = datetime.strptime(self.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(self.end_date, "%Y-%m-%d")
        while refresh_date <= end_dt:
            batch = self.loader.get_liquid_symbols(
                as_of_date=refresh_date.strftime("%Y-%m-%d"),
                min_avg_volume=200_000,
                min_price=5.0,
                max_symbols=2000,
            )
            liquid_symbols_set.update(batch)
            refresh_date += timedelta(days=90)
        liquid_symbols = sorted(liquid_symbols_set)
        logger.info("Liquid universe: %d symbols (cumulative)", len(liquid_symbols))

        # Load all daily data for the universe
        logger.info("Loading daily bars for universe (this may take a moment)...")
        all_daily = self.loader.load_daily(
            symbols=liquid_symbols,
            start_date=lookback_start,
            end_date=self.end_date,
            min_volume=0,
            min_price=0,
        )
        logger.info("Loaded %d rows for %d symbols", len(all_daily), all_daily["symbol"].nunique())

        # Build per-symbol daily DataFrames
        symbol_daily: Dict[str, pd.DataFrame] = {}
        for sym, grp in all_daily.groupby("symbol"):
            symbol_daily[str(sym)] = grp.reset_index(drop=True)

        # Build per-symbol weekly DataFrames
        logger.info("Aggregating to weekly bars...")
        symbol_weekly: Dict[str, pd.DataFrame] = {}
        for sym, df in symbol_daily.items():
            if len(df) >= 40:
                symbol_weekly[sym] = aggregate_to_weekly(df)

        # Generate trading days
        trade_start = pd.Timestamp(self.start_date)
        trade_end = pd.Timestamp(self.end_date)

        spy_dates = pd.to_datetime(spy_daily["date"]).values
        trading_days = sorted(set(
            pd.Timestamp(d).strftime("%Y-%m-%d")
            for d in spy_dates
            if trade_start <= pd.Timestamp(d) <= trade_end
        ))
        logger.info("Trading days: %d", len(trading_days))

        # Track weekly state
        current_week = None
        trades_this_week = 0
        candidates: List[dict] = []  # current week's candidates
        kill_switch_on_count = 0
        kill_switch_off_count = 0

        for day_idx, day_str in enumerate(trading_days):
            day_ts = pd.Timestamp(day_str)
            week_key = day_ts.isocalendar()[:2]  # (year, week)

            # --- New week: run weekly analysis ---
            if week_key != current_week:
                current_week = week_key
                trades_this_week = 0

                # Kill switch check
                spy_weekly_up_to = spy_weekly[spy_weekly["week_start"] <= day_ts]
                active, reason = self.kill_switch.is_active(spy_weekly_up_to)

                if not active:
                    kill_switch_off_count += 1
                    candidates = []
                    self.weekly_stage2_counts.append((day_str, 0))
                    continue
                else:
                    kill_switch_on_count += 1

                # Classify universe
                stage2_syms = []
                for sym, wk in symbol_weekly.items():
                    wk_up_to = wk[wk["week_start"] <= day_ts]
                    if len(wk_up_to) >= 34:
                        stage = self.classifier.classify(sym, wk_up_to)
                        if stage == "STAGE_2":
                            stage2_syms.append(sym)

                self.weekly_stage2_counts.append((day_str, len(stage2_syms)))

                # Detect tight bases among Stage 2 stocks
                candidates = []
                for sym in stage2_syms:
                    if sym not in symbol_daily:
                        continue
                    df = symbol_daily[sym]
                    df_up_to = df[pd.to_datetime(df["date"]) <= day_ts]
                    if len(df_up_to) < 40:
                        continue
                    setup = self.base_detector.detect(sym, df_up_to)
                    if setup is not None:
                        candidates.append(setup)

                # Rank by score, take top 15
                candidates.sort(key=lambda x: x["score"], reverse=True)
                candidates = candidates[:15]

            # --- Daily: manage positions ---
            self._manage_positions(day_str, symbol_daily)

            # --- Daily: look for entries ---
            if trades_this_week >= self.MAX_TRADES_PER_WEEK:
                continue
            if len(self.positions) >= self.MAX_POSITIONS:
                continue

            held_symbols = {p.symbol for p in self.positions}

            for cand in candidates:
                if trades_this_week >= self.MAX_TRADES_PER_WEEK:
                    break
                if len(self.positions) >= self.MAX_POSITIONS:
                    break

                sym = cand["symbol"]
                if sym in held_symbols:
                    continue
                if sym not in symbol_daily:
                    continue

                df = symbol_daily[sym]
                df_up_to = df[pd.to_datetime(df["date"]) <= day_ts]
                if len(df_up_to) < 50:
                    continue

                closes = df_up_to["close"].astype(float)
                last_close = closes.iloc[-1]

                # Check EMA zone: close in 9-21 EMA zone
                ema9 = _ema(closes, 9).iloc[-1]
                ema21 = _ema(closes, 21).iloc[-1]

                ema_low = min(ema9, ema21)
                ema_high = max(ema9, ema21)

                # Allow small buffer (1% below lower EMA)
                if not (ema_low * 0.99 <= last_close <= ema_high * 1.01):
                    continue

                # Volume check: pullback should be on low volume
                volumes = df_up_to["volume"].astype(float)
                if len(volumes) < 20:
                    continue
                vol_ratio = volumes.iloc[-1] / volumes.iloc[-20:].mean() if volumes.iloc[-20:].mean() > 0 else 999
                if vol_ratio > 0.80:
                    continue

                # Close > yesterday's low
                if len(df_up_to) < 2:
                    continue
                yesterday_low = float(df_up_to["low"].iloc[-2])
                if last_close < yesterday_low:
                    continue

                # Multi-timeframe bias
                sym_wk = symbol_weekly.get(sym)
                spy_wk_up_to = spy_weekly[spy_weekly["week_start"] <= day_ts]
                bias_result = self.bias.check_bias(sym, df_up_to, spy_wk_up_to, sym_wk)
                if not bias_result["all_pass"]:
                    continue

                # ENTRY — find next day's open
                next_day_idx = day_idx + 1
                if next_day_idx >= len(trading_days):
                    continue
                next_day = trading_days[next_day_idx]
                next_bars = df[pd.to_datetime(df["date"]) == pd.Timestamp(next_day)]
                if next_bars.empty:
                    continue

                entry_price = float(next_bars["open"].iloc[0])
                if entry_price <= 0:
                    continue

                # Stop = swing low of last 10 bars
                swing_low = float(df_up_to["low"].iloc[-10:].min())
                stop_price = swing_low * 0.995  # tiny buffer below swing low

                risk_per_share = entry_price - stop_price
                if risk_per_share <= 0:
                    continue
                if risk_per_share / entry_price > 0.08:
                    # Skip if stop is too wide (>8%)
                    continue

                # Position sizing
                risk_amount = self.equity * bias_result["risk_pct"]
                shares = int(risk_amount / risk_per_share)
                if shares <= 0:
                    continue

                # Check we can afford it
                cost = shares * entry_price
                if cost > self.equity * 0.25:
                    # Max 25% of equity in one position
                    shares = int(self.equity * 0.25 / entry_price)
                    if shares <= 0:
                        continue

                target_price = entry_price + 2 * risk_per_share

                pos = SwingPosition(
                    symbol=sym,
                    entry_date=next_day,
                    entry_price=entry_price,
                    shares=shares,
                    stop_price=stop_price,
                    target_price=target_price,
                    initial_risk=risk_per_share,
                )
                self.positions.append(pos)
                held_symbols.add(sym)
                trades_this_week += 1

            # Record equity
            self.equity_curve.append((day_str, self.equity))

        # Close any remaining positions at last available prices
        for pos in list(self.positions):
            self._close_position(pos, trading_days[-1], symbol_daily, "END_OF_BACKTEST")

        self._print_results()

    def _manage_positions(self, day_str: str, symbol_daily: Dict[str, pd.DataFrame]) -> None:
        """Manage open positions for the day."""
        day_ts = pd.Timestamp(day_str)
        to_close: List[Tuple[SwingPosition, str]] = []

        for pos in self.positions:
            df = symbol_daily.get(pos.symbol)
            if df is None:
                continue

            today_bars = df[pd.to_datetime(df["date"]) == day_ts]
            if today_bars.empty:
                continue

            today_low = float(today_bars["low"].iloc[0])
            today_close = float(today_bars["close"].iloc[0])
            today_high = float(today_bars["high"].iloc[0])

            pnl_per_share = today_close - pos.entry_price
            r_multiple = pnl_per_share / pos.initial_risk if pos.initial_risk > 0 else 0

            # Stop loss hit
            if today_low <= pos.stop_price:
                to_close.append((pos, "STOP_LOSS"))
                continue

            # Move stop to breakeven at 1R
            if r_multiple >= self.BREAKEVEN_R and not pos.stop_at_breakeven:
                pos.stop_price = pos.entry_price + 0.01  # tiny profit
                pos.stop_at_breakeven = True

            # Partial exit at 2R
            if r_multiple >= self.PARTIAL_EXIT_R and not pos.partial_exited:
                exit_shares = pos.shares // 2
                if exit_shares > 0:
                    pnl = exit_shares * (today_close - pos.entry_price)
                    self.equity += pnl
                    self.peak_equity = max(self.peak_equity, self.equity)
                    self.trades.append(SwingTrade(
                        symbol=pos.symbol,
                        entry_date=pos.entry_date,
                        exit_date=day_str,
                        entry_price=pos.entry_price,
                        exit_price=today_close,
                        shares=exit_shares,
                        pnl=pnl,
                        pnl_pct=pnl / (exit_shares * pos.entry_price) if pos.entry_price > 0 else 0,
                        r_multiple=r_multiple,
                        exit_reason="PARTIAL_2R",
                        holding_days=(pd.Timestamp(day_str) - pd.Timestamp(pos.entry_date)).days,
                    ))
                    pos.shares -= exit_shares
                    pos.partial_exited = True

            # Trail remaining below 21 EMA after partial
            if pos.partial_exited:
                df_up_to = df[pd.to_datetime(df["date"]) <= day_ts]
                if len(df_up_to) >= 21:
                    ema21_val = float(_ema(df_up_to["close"].astype(float), 21).iloc[-1])
                    # Trail stop is max of current stop and 21 EMA - 0.5%
                    new_stop = ema21_val * 0.995
                    if new_stop > pos.stop_price:
                        pos.stop_price = new_stop

            # Exit if close < 50 EMA
            df_up_to = df[pd.to_datetime(df["date"]) <= day_ts]
            if len(df_up_to) >= 50:
                ema50_val = float(_ema(df_up_to["close"].astype(float), 50).iloc[-1])
                if today_close < ema50_val:
                    to_close.append((pos, "BELOW_50_EMA"))
                    continue

            # Exit if 3 consecutive non-higher closes with declining volume
            if len(df_up_to) >= 4:
                recent = df_up_to.iloc[-4:]
                recent_closes = recent["close"].astype(float).values
                recent_vols = recent["volume"].astype(float).values
                non_higher = all(recent_closes[i] <= recent_closes[i - 1] for i in range(1, 4))
                vol_declining = all(recent_vols[i] <= recent_vols[i - 1] for i in range(1, 4))
                if non_higher and vol_declining:
                    to_close.append((pos, "STALL_EXIT"))
                    continue

        for pos, reason in to_close:
            self._close_position(pos, day_str, symbol_daily, reason)

    def _close_position(
        self,
        pos: SwingPosition,
        day_str: str,
        symbol_daily: Dict[str, pd.DataFrame],
        reason: str,
    ) -> None:
        """Close a position and record the trade."""
        df = symbol_daily.get(pos.symbol)
        exit_price = pos.entry_price  # fallback

        if df is not None:
            day_ts = pd.Timestamp(day_str)
            today = df[pd.to_datetime(df["date"]) == day_ts]
            if not today.empty:
                if reason == "STOP_LOSS":
                    exit_price = pos.stop_price
                else:
                    exit_price = float(today["close"].iloc[0])

        pnl = pos.shares * (exit_price - pos.entry_price)
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)

        r_mult = (exit_price - pos.entry_price) / pos.initial_risk if pos.initial_risk > 0 else 0

        self.trades.append(SwingTrade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=day_str,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl / (pos.shares * pos.entry_price) if pos.entry_price > 0 else 0,
            r_multiple=r_mult,
            exit_reason=reason,
            holding_days=(pd.Timestamp(day_str) - pd.Timestamp(pos.entry_date)).days,
        ))

        if pos in self.positions:
            self.positions.remove(pos)

    def _print_results(self) -> None:
        """Print comprehensive backtest results."""
        print("\n" + "=" * 80)
        print("  RUDRA SWING ENGINE — BACKTEST RESULTS")
        print("=" * 80)
        print(f"  Period:          {self.start_date} to {self.end_date}")
        print(f"  Initial Capital: ${self.initial_capital:,.0f}")
        print(f"  Final Equity:    ${self.equity:,.0f}")
        print(f"  Total Return:    {(self.equity / self.initial_capital - 1) * 100:.1f}%")

        if not self.trades:
            print("\n  NO TRADES GENERATED.")
            print("=" * 80)
            return

        total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        net_pnl = sum(t.pnl for t in self.trades)

        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        avg_r = sum(t.r_multiple for t in self.trades) / total_trades
        avg_hold = sum(t.holding_days for t in self.trades) / total_trades

        # Max drawdown
        max_dd = 0
        peak = self.initial_capital
        for _, eq in self.equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Sharpe (annualized, using daily equity changes)
        if len(self.equity_curve) > 1:
            eqs = [e for _, e in self.equity_curve]
            returns = [(eqs[i] - eqs[i - 1]) / eqs[i - 1] for i in range(1, len(eqs)) if eqs[i - 1] > 0]
            if returns:
                avg_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = (avg_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        print(f"\n  --- Performance ---")
        print(f"  Total Trades:    {total_trades}")
        print(f"  Wins / Losses:   {len(wins)} / {len(losses)}")
        print(f"  Win Rate:        {win_rate:.1f}%")
        print(f"  Profit Factor:   {pf:.2f}")
        print(f"  Net P&L:         ${net_pnl:,.0f}")
        print(f"  Avg Win:         ${avg_win:,.0f}")
        print(f"  Avg Loss:        ${avg_loss:,.0f}")
        print(f"  Avg R-Multiple:  {avg_r:.2f}R")
        print(f"  Avg Hold Days:   {avg_hold:.1f}")
        print(f"  Max Drawdown:    {max_dd * 100:.1f}%")
        print(f"  Sharpe Ratio:    {sharpe:.2f}")

        # Monthly breakdown
        print(f"\n  --- Monthly Breakdown ---")
        monthly: Dict[str, List[SwingTrade]] = {}
        for t in self.trades:
            month_key = t.exit_date[:7]
            monthly.setdefault(month_key, []).append(t)

        print(f"  {'Month':<10} {'Trades':>7} {'Win%':>7} {'PnL':>10} {'Avg R':>7}")
        print(f"  {'-'*41}")
        for month in sorted(monthly.keys()):
            mt = monthly[month]
            m_wins = sum(1 for t in mt if t.pnl > 0)
            m_wr = m_wins / len(mt) * 100 if mt else 0
            m_pnl = sum(t.pnl for t in mt)
            m_avg_r = sum(t.r_multiple for t in mt) / len(mt) if mt else 0
            print(f"  {month:<10} {len(mt):>7} {m_wr:>6.0f}% ${m_pnl:>9,.0f} {m_avg_r:>6.2f}R")

        # Yearly breakdown
        print(f"\n  --- Yearly Breakdown ---")
        yearly: Dict[str, List[SwingTrade]] = {}
        for t in self.trades:
            yearly.setdefault(t.exit_date[:4], []).append(t)

        print(f"  {'Year':<6} {'Trades':>7} {'Win%':>7} {'PnL':>10} {'PF':>7}")
        print(f"  {'-'*37}")
        for year in sorted(yearly.keys()):
            yt = yearly[year]
            y_wins = sum(1 for t in yt if t.pnl > 0)
            y_wr = y_wins / len(yt) * 100 if yt else 0
            y_pnl = sum(t.pnl for t in yt)
            y_gp = sum(t.pnl for t in yt if t.pnl > 0)
            y_gl = abs(sum(t.pnl for t in yt if t.pnl <= 0))
            y_pf = y_gp / y_gl if y_gl > 0 else float("inf")
            print(f"  {year:<6} {len(yt):>7} {y_wr:>6.0f}% ${y_pnl:>9,.0f} {y_pf:>6.2f}")

        # Weekly Stage 2 counts
        if self.weekly_stage2_counts:
            avg_s2 = np.mean([c for _, c in self.weekly_stage2_counts])
            max_s2 = max(c for _, c in self.weekly_stage2_counts)
            min_s2 = min(c for _, c in self.weekly_stage2_counts)
            print(f"\n  --- Stage 2 Universe ---")
            print(f"  Avg Stage 2 stocks/week: {avg_s2:.0f}")
            print(f"  Max Stage 2 stocks/week: {max_s2}")
            print(f"  Min Stage 2 stocks/week: {min_s2}")
            print(f"  Kill switch OFF weeks:   {self.kill_switch._was_halted}")

        # Top winners and losers
        sorted_trades = sorted(self.trades, key=lambda t: t.pnl, reverse=True)
        print(f"\n  --- Top 5 Winners ---")
        print(f"  {'Symbol':<8} {'Entry':<12} {'Exit':<12} {'PnL':>10} {'R':>7} {'Reason'}")
        for t in sorted_trades[:5]:
            print(f"  {t.symbol:<8} {t.entry_date:<12} {t.exit_date:<12} ${t.pnl:>9,.0f} {t.r_multiple:>6.2f}R {t.exit_reason}")

        print(f"\n  --- Top 5 Losers ---")
        for t in sorted_trades[-5:]:
            print(f"  {t.symbol:<8} {t.entry_date:<12} {t.exit_date:<12} ${t.pnl:>9,.0f} {t.r_multiple:>6.2f}R {t.exit_reason}")

        # Exit reason breakdown
        print(f"\n  --- Exit Reasons ---")
        reason_counts: Dict[str, int] = {}
        reason_pnl: Dict[str, float] = {}
        for t in self.trades:
            reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1
            reason_pnl[t.exit_reason] = reason_pnl.get(t.exit_reason, 0) + t.pnl
        for reason in sorted(reason_counts.keys()):
            print(f"  {reason:<20} {reason_counts[reason]:>5} trades  ${reason_pnl[reason]:>10,.0f}")

        print("\n" + "=" * 80)


# ---------------------------------------------------------------------------
# CLI: Scan Mode
# ---------------------------------------------------------------------------

def run_scan(db_url: str = DB_URL) -> None:
    """Run a current scan for Stage 2 stocks with tight bases."""
    loader = DataLoader(db_url)
    kill_switch = WeeklyKillSwitch()
    classifier = StageClassifier()
    base_detector = TightBaseDetector()
    volume_engine = VolumeEngine()

    today = date.today().strftime("%Y-%m-%d")
    lookback_start = (date.today() - timedelta(days=400)).strftime("%Y-%m-%d")

    # SPY kill switch
    spy_daily = loader.load_daily(symbols=["SPY"], start_date=lookback_start, end_date=today)
    spy_weekly = aggregate_to_weekly(spy_daily)
    active, reason = kill_switch.is_active(spy_weekly)
    print(f"\n{'='*60}")
    print(f"  WEEKLY KILL SWITCH: {'ACTIVE (trading allowed)' if active else 'OFF (no new entries)'}")
    print(f"  {reason}")
    print(f"{'='*60}")

    if not active:
        print("\n  System halted. No scan performed.\n")
        return

    # Get liquid symbols
    symbols = loader.get_liquid_symbols(as_of_date=today, min_avg_volume=500_000, min_price=10.0, max_symbols=1500)
    print(f"\n  Scanning {len(symbols)} liquid symbols...")

    # Load data
    all_daily = loader.load_daily(symbols=symbols, start_date=lookback_start, end_date=today, min_volume=0, min_price=0)

    # Classify
    stage2_syms = []
    stage_counts = {"STAGE_1": 0, "STAGE_2": 0, "STAGE_3": 0, "STAGE_4": 0, "UNCLEAR": 0}

    for sym, grp in all_daily.groupby("symbol"):
        weekly = aggregate_to_weekly(grp.reset_index(drop=True))
        if len(weekly) >= 34:
            stage = classifier.classify(str(sym), weekly)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            if stage == "STAGE_2":
                stage2_syms.append(str(sym))

    print(f"\n  Stage Distribution:")
    for stage, count in sorted(stage_counts.items()):
        print(f"    {stage}: {count}")
    print(f"\n  Stage 2 stocks: {len(stage2_syms)}")

    # Detect tight bases
    setups = []
    for sym in stage2_syms:
        sym_daily = all_daily[all_daily["symbol"] == sym].reset_index(drop=True)
        setup = base_detector.detect(sym, sym_daily)
        if setup is not None:
            # Add volume state
            vol_state = volume_engine.classify(sym, sym_daily)
            setup["vol_state"] = vol_state
            setups.append(setup)

    setups.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n  Tight Base Setups Found: {len(setups)}")
    if setups:
        print(f"\n  {'Sym':<8} {'Score':>6} {'Comp':>6} {'VolR':>6} {'Move%':>6} {'Days':>5} {'Vol State'}")
        print(f"  {'-'*47}")
        for s in setups[:20]:
            print(
                f"  {s['symbol']:<8} {s['score']:>5.1f} {s['compression_ratio']:>5.2f} "
                f"{s['volume_ratio']:>5.2f} {s['prior_move_pct']*100:>5.1f}% {s['days_in_base']:>5} "
                f"{s.get('vol_state', 'N/A')}"
            )
    print()


# ---------------------------------------------------------------------------
# CLI: Classify Mode
# ---------------------------------------------------------------------------

def run_classify(symbol: str, db_url: str = DB_URL) -> None:
    """Classify a single stock."""
    loader = DataLoader(db_url)
    kill_switch = WeeklyKillSwitch()
    classifier = StageClassifier()
    volume_engine = VolumeEngine()
    bias = MultiTimeframeBias(kill_switch)

    today = date.today().strftime("%Y-%m-%d")
    lookback_start = (date.today() - timedelta(days=600)).strftime("%Y-%m-%d")

    sym_daily = loader.load_daily(symbols=[symbol], start_date=lookback_start, end_date=today)
    spy_daily = loader.load_daily(symbols=["SPY"], start_date=lookback_start, end_date=today)

    if sym_daily.empty:
        print(f"\n  No data found for {symbol}\n")
        return

    sym_weekly = aggregate_to_weekly(sym_daily)
    spy_weekly = aggregate_to_weekly(spy_daily)

    stage = classifier.classify(symbol, sym_weekly)
    vol_state = volume_engine.classify(symbol, sym_daily)
    trend = volume_engine.trend_health(symbol, sym_daily)
    bias_result = bias.check_bias(symbol, sym_daily, spy_weekly, sym_weekly)

    last_close = float(sym_daily["close"].iloc[-1])
    last_date = sym_daily["date"].iloc[-1]

    print(f"\n{'='*60}")
    print(f"  {symbol} Classification — as of {last_date}")
    print(f"{'='*60}")
    print(f"  Last Close:       ${last_close:.2f}")
    print(f"  Weinstein Stage:  {stage}")
    print(f"  Volume State:     {vol_state}")
    print(f"  Trend Health:     {trend}")
    print(f"\n  --- Multi-Timeframe Bias ---")
    print(f"  System Active:    {bias_result['system_active']}")
    print(f"  Stage 2:          {bias_result['stage_2']}")
    print(f"  Above 200 EMA:    {bias_result['above_200']}")
    print(f"  EMA Aligned:      {bias_result['ema_aligned']}")
    print(f"  All Pass:         {bias_result['all_pass']}")
    print(f"  Risk %:           {bias_result['risk_pct']*100:.1f}%")
    print(f"  Details:          {bias_result['details']}")

    # Tight base?
    base_detector = TightBaseDetector()
    setup = base_detector.detect(symbol, sym_daily)
    if setup:
        print(f"\n  --- Tight Base Detected ---")
        print(f"  Score:            {setup['score']:.1f}")
        print(f"  Base High:        ${setup['base_high']:.2f}")
        print(f"  Base Low:         ${setup['base_low']:.2f}")
        print(f"  Compression:      {setup['compression_ratio']:.2f}")
        print(f"  Volume Ratio:     {setup['volume_ratio']:.2f}")
        print(f"  Prior Move:       {setup['prior_move_pct']*100:.1f}%")
        print(f"  Days in Base:     {setup['days_in_base']}")
    else:
        print(f"\n  No tight base detected.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Rudra Swing Engine — Weinstein Stage Analysis backtester")
    parser.add_argument("--scan", action="store_true", help="Run current weekly scan")
    parser.add_argument("--classify", type=str, metavar="SYMBOL", help="Classify a single stock")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Backtest start date")
    parser.add_argument("--end", type=str, default="2026-03-21", help="Backtest end date")
    parser.add_argument("--capital", type=float, default=25_000.0, help="Initial capital")
    parser.add_argument("--db-url", type=str, default=DB_URL, help="PostgreSQL connection URL")

    args = parser.parse_args()

    if args.scan:
        run_scan(args.db_url)
    elif args.classify:
        run_classify(args.classify.upper(), args.db_url)
    else:
        bt = SwingBacktester(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            db_url=args.db_url,
        )
        bt.run()


if __name__ == "__main__":
    main()
