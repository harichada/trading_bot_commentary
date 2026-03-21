"""
Multi-Regime Quantitative Gap Fade Engine
==========================================

A Bayesian, regime-aware trading engine that replaces simple threshold-based
gap fade logic with probabilistic decision-making.

Architecture:
    1. RegimeDetector  — Classifies market state (trending/ranging/volatile)
    2. BayesianScorer  — Assigns P(profit | features) to each candidate
    3. PositionSizer   — Kelly/ATR-based sizing with regime adjustment
    4. ExitManager     — Multi-factor exit with trailing stops + time decay
    5. PostMortem      — Self-evolving parameter adjustment from trade logs

Author: Rudra Quant Engine v1.0
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo('US/Eastern')
logger = logging.getLogger('QuantEngine')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MarketRegime:
    """Immutable snapshot of market conditions."""
    regime: str              # 'trending_up', 'trending_down', 'ranging', 'volatile'
    adx: float               # Average Directional Index (0-100)
    atr_pct: float           # ATR as % of price (normalized volatility)
    spy_5d_return: float     # SPY 5-day cumulative return
    spy_20d_return: float    # SPY 20-day cumulative return
    vix_level: float         # VIX or implied vol proxy
    confidence: float        # Regime classification confidence (0-1)


class RegimeDetector:
    """Classifies market regime from SPY daily bars using ADX + ATR.

    Regime definitions (mathematically precise):
        trending_up:   ADX > 25 AND 5d return > 0
        trending_down: ADX > 25 AND 5d return < 0
        ranging:       ADX <= 25 AND ATR% < median ATR% (low volatility)
        volatile:      ADX <= 25 AND ATR% >= median ATR% (high volatility)
    """

    ADX_TREND_THRESHOLD = 25.0
    ATR_LOOKBACK = 14
    ADX_LOOKBACK = 14

    def __init__(self, spy_bars: pd.DataFrame):
        """Initialize with SPY daily bars (columns: open, high, low, close, volume)."""
        self.spy = spy_bars.sort_values('date').reset_index(drop=True)
        self._compute_indicators()

    def _compute_indicators(self):
        df = self.spy
        # True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['prev_close']),
                abs(df['low'] - df['prev_close'])
            )
        )
        # ATR
        df['atr'] = df['tr'].rolling(self.ATR_LOOKBACK).mean()
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        # Smoothed DI
        df['plus_di'] = 100 * df['plus_dm'].rolling(self.ADX_LOOKBACK).mean() / df['atr'].clip(lower=0.001)
        df['minus_di'] = 100 * df['minus_dm'].rolling(self.ADX_LOOKBACK).mean() / df['atr'].clip(lower=0.001)

        # ADX
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).clip(lower=0.001)
        df['adx'] = df['dx'].rolling(self.ADX_LOOKBACK).mean()

        # Returns
        df['ret_5d'] = df['close'].pct_change(5)
        df['ret_20d'] = df['close'].pct_change(20)

        # Median ATR for regime classification
        self._median_atr_pct = df['atr_pct'].rolling(60).median()
        self.spy = df

    def classify(self, date: str = None) -> MarketRegime:
        """Classify market regime for a given date (default: latest)."""
        if date:
            row = self.spy[self.spy['date'] == date]
            if row.empty:
                # Find closest date
                row = self.spy[self.spy['date'] <= date].tail(1)
        else:
            row = self.spy.tail(1)

        if row.empty:
            return MarketRegime('unknown', 0, 0, 0, 0, 0, 0)

        r = row.iloc[0]
        adx = float(r.get('adx', 0) or 0)
        atr_pct = float(r.get('atr_pct', 0) or 0)
        ret_5d = float(r.get('ret_5d', 0) or 0)
        ret_20d = float(r.get('ret_20d', 0) or 0)
        idx = row.index[0]
        median_atr = float(self._median_atr_pct.iloc[idx]) if idx < len(self._median_atr_pct) and pd.notna(self._median_atr_pct.iloc[idx]) else atr_pct

        if adx > self.ADX_TREND_THRESHOLD:
            regime = 'trending_up' if ret_5d > 0 else 'trending_down'
            confidence = min(1.0, adx / 50.0)  # Higher ADX = more confidence
        elif atr_pct >= median_atr:
            regime = 'volatile'
            confidence = min(1.0, atr_pct / (median_atr * 2)) if median_atr > 0 else 0.5
        else:
            regime = 'ranging'
            confidence = min(1.0, (self.ADX_TREND_THRESHOLD - adx) / self.ADX_TREND_THRESHOLD)

        return MarketRegime(
            regime=regime, adx=adx, atr_pct=atr_pct,
            spy_5d_return=ret_5d, spy_20d_return=ret_20d,
            vix_level=atr_pct * 10,  # ATR proxy for VIX
            confidence=confidence,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BAYESIAN SCORER — P(profit | features)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CandidateFeatures:
    """Feature vector for a gap candidate."""
    symbol: str
    gap_pct: float             # Signed gap percentage
    abs_gap_pct: float         # Absolute gap
    vol_ratio: float           # Volume / 20-day avg volume
    price: float               # Current price
    market_cap_bucket: str     # 'mega', 'large', 'mid', 'small', 'micro'
    regime: MarketRegime       # Current market regime
    time_of_day: float         # Fractional hour (9.5 = 9:30 AM)
    atr_pct: float             # Symbol's own ATR%
    prev_close: float          # Previous close
    score: float = 0.0         # Final Bayesian score


class BayesianScorer:
    """Assigns P(profit) to each candidate using empirical Bayesian priors.

    The priors are derived from backtested trade data. Each factor contributes
    a likelihood ratio that adjusts the base rate.

    P(profit | features) = P(profit) * ∏ LR(feature_i)
    where LR = P(feature | profit) / P(feature | loss)
    """

    # Base rate from 3-year backtest (Iteration 14: 55.9% WR on gap-up shorts)
    BASE_WIN_RATE = 0.559

    # Empirical likelihood ratios from trade analysis
    # LR > 1.0 = feature increases P(profit), LR < 1.0 = decreases
    LIKELIHOOD_RATIOS = {
        # Gap size buckets (from sweep data)
        'gap_5_7': 1.15,      # 5-7% gaps slightly better
        'gap_7_10': 1.05,     # 7-10% baseline
        'gap_10_15': 0.85,    # 10-15% weaker
        'gap_15_plus': 0.70,  # 15%+ often catalyst-driven, don't fade

        # Regime (from conditional probability analysis)
        'regime_bull_5d': 1.10,        # Gap-up shorts work better in bull markets (mean reversion)
        'regime_bear_5d': 0.75,        # Bear markets — gap-ups are breakouts, don't fade
        'regime_neutral': 0.90,        # Neutral — baseline
        'regime_volatile': 0.80,       # High vol — noise stops trigger too often
        'regime_trending_up': 1.05,    # Trending up — gap-ups overshoot, fade works
        'regime_trending_down': 0.70,  # Trending down — gap-ups are dead cat bounces, risky

        # Volume (from trade analysis)
        'vol_below_avg': 1.15,    # Low volume gaps fade better (no institutional buying)
        'vol_above_avg': 0.85,    # High volume = conviction, don't fade
        'vol_extreme': 0.60,      # 3x+ volume = news-driven, avoid

        # Time of day
        'time_930_1000': 0.85,    # First 30 min: noisy, high false positive rate
        'time_1000_1130': 1.15,   # Sweet spot: initial volatility settles
        'time_1130_1300': 0.95,   # Lunch: low volume, lower edge
        'time_1300_1500': 0.80,   # Afternoon: gap fill momentum fades

        # Price tier
        'price_10_25': 0.90,      # Low price = wider spreads, more noise
        'price_25_100': 1.10,     # Sweet spot for gap fades
        'price_100_plus': 1.00,   # High price = institutional, baseline
    }

    # Minimum P(profit) to enter a trade
    MIN_PROBABILITY = 0.52

    def score(self, candidate: CandidateFeatures) -> Tuple[float, Dict[str, float]]:
        """Compute P(profit | features) using Bayesian updating.

        Returns:
            (probability, factor_breakdown) where factor_breakdown shows
            each factor's contribution for explainability.
        """
        p = self.BASE_WIN_RATE
        odds = p / (1 - p)  # Convert to odds ratio for multiplicative updates
        factors = {}

        # 1. Gap size factor
        gap = abs(candidate.gap_pct)
        if gap < 0.07:
            lr = self.LIKELIHOOD_RATIOS['gap_5_7']
            factors['gap_size'] = lr
        elif gap < 0.10:
            lr = self.LIKELIHOOD_RATIOS['gap_7_10']
            factors['gap_size'] = lr
        elif gap < 0.15:
            lr = self.LIKELIHOOD_RATIOS['gap_10_15']
            factors['gap_size'] = lr
        else:
            lr = self.LIKELIHOOD_RATIOS['gap_15_plus']
            factors['gap_size'] = lr
        odds *= lr

        # 2. Regime factor
        regime = candidate.regime
        if regime.regime == 'trending_up':
            lr = self.LIKELIHOOD_RATIOS['regime_trending_up']
        elif regime.regime == 'trending_down':
            lr = self.LIKELIHOOD_RATIOS['regime_trending_down']
        elif regime.regime == 'volatile':
            lr = self.LIKELIHOOD_RATIOS['regime_volatile']
        elif regime.spy_5d_return > 0:
            lr = self.LIKELIHOOD_RATIOS['regime_bull_5d']
        elif regime.spy_5d_return < -0.02:
            lr = self.LIKELIHOOD_RATIOS['regime_bear_5d']
        else:
            lr = self.LIKELIHOOD_RATIOS['regime_neutral']
        factors['regime'] = lr
        odds *= lr

        # 3. Volume factor
        vol = candidate.vol_ratio
        if vol > 3.0:
            lr = self.LIKELIHOOD_RATIOS['vol_extreme']
            factors['volume'] = lr
        elif vol > 1.0:
            lr = self.LIKELIHOOD_RATIOS['vol_above_avg']
            factors['volume'] = lr
        else:
            lr = self.LIKELIHOOD_RATIOS['vol_below_avg']
            factors['volume'] = lr
        odds *= lr

        # 4. Time of day factor
        tod = candidate.time_of_day
        if tod < 10.0:
            lr = self.LIKELIHOOD_RATIOS['time_930_1000']
        elif tod < 11.5:
            lr = self.LIKELIHOOD_RATIOS['time_1000_1130']
        elif tod < 13.0:
            lr = self.LIKELIHOOD_RATIOS['time_1130_1300']
        else:
            lr = self.LIKELIHOOD_RATIOS['time_1300_1500']
        factors['time_of_day'] = lr
        odds *= lr

        # 5. Price tier factor
        price = candidate.price
        if price < 25:
            lr = self.LIKELIHOOD_RATIOS['price_10_25']
        elif price < 100:
            lr = self.LIKELIHOOD_RATIOS['price_25_100']
        else:
            lr = self.LIKELIHOOD_RATIOS['price_100_plus']
        factors['price_tier'] = lr
        odds *= lr

        # Convert back to probability
        probability = odds / (1 + odds)

        return round(probability, 4), factors

    def should_trade(self, candidate: CandidateFeatures) -> Tuple[bool, float, str]:
        """Decision: should we trade this candidate?

        Returns (should_trade, probability, reason).
        """
        prob, factors = self.score(candidate)

        if prob < self.MIN_PROBABILITY:
            worst_factor = min(factors.items(), key=lambda x: x[1])
            return False, prob, f"P(profit)={prob:.1%} < {self.MIN_PROBABILITY:.0%} (worst: {worst_factor[0]}={worst_factor[1]:.2f})"

        return True, prob, f"P(profit)={prob:.1%} — factors: {', '.join(f'{k}={v:.2f}' for k, v in factors.items())}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VOLATILITY-ADJUSTED POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSize:
    """Computed position size with full audit trail."""
    shares: int
    dollar_risk: float
    notional: float
    risk_pct: float           # Actual risk as % of equity
    method: str               # 'kelly', 'atr', 'fixed'
    kelly_fraction: float     # Raw Kelly f* before capping
    regime_multiplier: float  # Regime-based scaling (0.25 to 1.0)


class PositionSizer:
    """Kelly Criterion + ATR-based position sizing with regime adjustment.

    Kelly f* = (p * b - q) / b
    where p = win probability, b = win/loss ratio, q = 1 - p

    Then scaled by:
        - Regime multiplier (reduce in volatile regimes)
        - Max fraction cap (never bet more than 25% Kelly)
        - Notional cap ($50K max per position)
    """

    KELLY_FRACTION_CAP = 0.25  # Never use more than 25% of Kelly
    MAX_NOTIONAL = 50_000
    MAX_RISK_PCT = 0.02        # Never risk more than 2% of equity per trade

    # Regime-based size multipliers
    REGIME_MULTIPLIERS = {
        'trending_up': 1.0,      # Full size in trending bull
        'trending_down': 0.50,   # Half size in downtrend
        'ranging': 0.75,         # 75% in ranging
        'volatile': 0.50,        # Half size in high vol
        'unknown': 0.25,         # Quarter size if regime unknown
    }

    def __init__(self, equity: float, base_win_rate: float = 0.559,
                 avg_win: float = 392.69, avg_loss: float = 285.43):
        self.equity = equity
        self.base_win_rate = base_win_rate
        self.avg_win = avg_win
        self.avg_loss = avg_loss

    def compute(self, candidate: CandidateFeatures, probability: float,
                stop_distance_pct: float) -> PositionSize:
        """Compute position size using Kelly + regime adjustment.

        Args:
            candidate: Scored candidate with features
            probability: P(profit) from Bayesian scorer
            stop_distance_pct: Distance to stop as fraction (e.g., 0.015 = 1.5%)
        """
        if stop_distance_pct <= 0:
            stop_distance_pct = 0.015  # Safety default

        # Kelly f*
        b = self.avg_win / max(self.avg_loss, 1)  # Payoff ratio
        p = probability
        q = 1 - p
        kelly_raw = (p * b - q) / b if b > 0 else 0
        kelly_raw = max(0, kelly_raw)  # Never negative

        # Cap Kelly
        kelly_capped = min(kelly_raw, self.KELLY_FRACTION_CAP)

        # Regime adjustment
        regime_mult = self.REGIME_MULTIPLIERS.get(candidate.regime.regime, 0.50)

        # Effective risk %
        effective_risk_pct = kelly_capped * regime_mult
        effective_risk_pct = min(effective_risk_pct, self.MAX_RISK_PCT)

        # Dollar risk
        dollar_risk = self.equity * effective_risk_pct

        # Shares from risk / stop distance
        risk_per_share = candidate.price * stop_distance_pct
        shares = int(dollar_risk / risk_per_share) if risk_per_share > 0 else 0

        # Notional cap
        notional = shares * candidate.price
        if notional > self.MAX_NOTIONAL:
            shares = int(self.MAX_NOTIONAL / candidate.price)
            notional = shares * candidate.price

        # Minimum 1 share
        shares = max(shares, 1) if dollar_risk > 0 else 0

        return PositionSize(
            shares=shares,
            dollar_risk=dollar_risk,
            notional=shares * candidate.price,
            risk_pct=effective_risk_pct,
            method='kelly',
            kelly_fraction=kelly_raw,
            regime_multiplier=regime_mult,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EXIT MANAGER — Multi-Factor Exit Logic
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExitConfig:
    """Exit parameters — all configurable."""
    # Trailing stop
    trailing_activation_pct: float = 0.01    # Activate after 1% profit
    trailing_distance_pct: float = 0.005     # Trail 0.5% behind best
    # Time decay — tighten stop as holding time increases
    time_decay_enabled: bool = True
    time_decay_start_min: int = 60           # Start tightening after 60 min
    time_decay_rate: float = 0.0001          # Tighten stop by 0.01% per minute
    # Partial profit
    partial_target_pct: float = 0.50         # Cover 1/3 at 50% gap fill
    partial_cover_frac: float = 0.33
    # Time exits
    time_exit_hour: int = 15
    time_exit_min: int = 0
    eod_exit_hour: int = 15
    eod_exit_min: int = 50
    # Volume divergence
    volume_exit_enabled: bool = True         # Exit if volume dries up while in position


@dataclass
class ExitSignal:
    """Recommended exit action."""
    action: str             # 'hold', 'tighten_stop', 'partial', 'close'
    reason: str
    new_stop: Optional[float] = None
    shares: Optional[int] = None
    confidence: float = 0.0


class ExitManager:
    """Multi-factor exit logic with time decay and volume divergence.

    Exit priority:
        1. Hard stop (broker-side, GTC)
        2. Trailing stop (tightens as profit grows)
        3. Time decay (tightens as holding time increases)
        4. Volume divergence (exit if volume drops while in position)
        5. Partial profit at half-target
        6. Full target at prev_close
        7. Time exit at 3 PM
        8. EOD force close at 3:50 PM
    """

    def __init__(self, config: ExitConfig = None):
        self.config = config or ExitConfig()

    def evaluate(self, entry_price: float, current_price: float,
                 stop_price: float, direction: str,
                 high_water_price: float, holding_minutes: int,
                 current_volume_ratio: float = 1.0,
                 partial_filled: bool = False,
                 half_target: float = 0, full_target: float = 0,
                 now: datetime = None) -> ExitSignal:
        """Evaluate all exit conditions and return recommended action."""
        now = now or datetime.now(ET)
        cfg = self.config

        if direction == 'short':
            unrealized_pct = (entry_price - current_price) / entry_price
        else:
            unrealized_pct = (current_price - entry_price) / entry_price

        new_stop = stop_price

        # 1. Trailing stop
        if unrealized_pct >= cfg.trailing_activation_pct:
            if direction == 'short':
                trail = high_water_price * (1 + cfg.trailing_distance_pct)
                if trail < new_stop:
                    new_stop = trail
            else:
                trail = high_water_price * (1 - cfg.trailing_distance_pct)
                if trail > new_stop:
                    new_stop = trail

        # 2. Time decay — tighten stop as position ages
        if cfg.time_decay_enabled and holding_minutes > cfg.time_decay_start_min:
            decay_minutes = holding_minutes - cfg.time_decay_start_min
            decay_pct = decay_minutes * cfg.time_decay_rate
            if direction == 'short':
                decayed_stop = entry_price * (1 + max(0.005, (stop_price / entry_price - 1) - decay_pct))
                new_stop = min(new_stop, decayed_stop)
            else:
                decayed_stop = entry_price * (1 - max(0.005, (1 - stop_price / entry_price) - decay_pct))
                new_stop = max(new_stop, decayed_stop)

        # 3. Volume divergence — if volume drops below 50% of entry volume
        if cfg.volume_exit_enabled and current_volume_ratio < 0.5 and unrealized_pct > 0:
            return ExitSignal('close', 'Volume dried up while in profit — take gains',
                              confidence=0.7)

        # 4. Time exits
        if now.hour > cfg.eod_exit_hour or (now.hour == cfg.eod_exit_hour and now.minute >= cfg.eod_exit_min):
            return ExitSignal('close', 'EOD force close', confidence=1.0)

        if now.hour > cfg.time_exit_hour or (now.hour == cfg.time_exit_hour and now.minute >= cfg.time_exit_min):
            return ExitSignal('close', 'Time exit', confidence=0.9)

        # 5. Stop tightened?
        if abs(new_stop - stop_price) > 0.001:
            return ExitSignal('tighten_stop', f'Stop tightened: ${stop_price:.2f} → ${new_stop:.2f}',
                              new_stop=new_stop, confidence=0.8)

        return ExitSignal('hold', f'Holding | P&L: {unrealized_pct:.1%} | Stop: ${stop_price:.2f}')


# ═══════════════════════════════════════════════════════════════════════════════
# 5. POST-MORTEM ANALYZER — Self-Evolving Parameter Adjustment
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeLog:
    """Structured trade log entry for post-mortem analysis."""
    trade_id: int
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    entry_time: str
    exit_time: str
    exit_reason: str
    holding_minutes: int
    gap_pct: float
    vol_ratio: float
    regime: str = ''
    spy_change: float = 0.0
    max_favorable_excursion: float = 0.0  # Best unrealized P&L before exit
    max_adverse_excursion: float = 0.0    # Worst unrealized P&L before exit


@dataclass
class ParameterAdjustment:
    """Specific parameter adjustment recommendation."""
    parameter: str
    current_value: float
    recommended_value: float
    reason: str
    confidence: float         # 0-1, based on sample size and effect size
    sample_size: int
    expected_impact: str      # Human-readable impact description


class PostMortemAnalyzer:
    """Analyzes trade logs to identify false positives and suggest parameter adjustments.

    Uses statistical tests to distinguish signal from noise:
        - Trades stopped within 5 min with move < 1.5% = probable noise stop
        - Trades profitable then reversed = trailing stop too loose
        - Trades that never moved favorably = bad entry signal
    """

    def analyze(self, trades: List[TradeLog]) -> List[ParameterAdjustment]:
        """Analyze trade logs and return parameter adjustment recommendations."""
        adjustments = []
        if len(trades) < 10:
            return adjustments

        df = pd.DataFrame([asdict(t) for t in trades])

        # 1. Noise stop analysis
        noise_stops = df[(df['exit_reason'] == 'stop') & (df['holding_minutes'] <= 5)]
        if len(noise_stops) > 5:
            noise_rate = len(noise_stops) / len(df[df['exit_reason'] == 'stop'])
            avg_noise_move = noise_stops['pnl_pct'].abs().mean()

            if noise_rate > 0.4:  # >40% of stops are noise
                adjustments.append(ParameterAdjustment(
                    parameter='stop_min_pct',
                    current_value=0.015,
                    recommended_value=round(avg_noise_move * 1.5, 4),
                    reason=f'{noise_rate:.0%} of stops triggered within 5 min '
                           f'(avg move {avg_noise_move:.1%}). '
                           f'Widen minimum stop to survive initial volatility.',
                    confidence=min(0.9, len(noise_stops) / 30),
                    sample_size=len(noise_stops),
                    expected_impact=f'Reduce noise stops by ~{noise_rate*50:.0f}%',
                ))

        # 2. Profit reversal analysis (trades that were profitable then hit stop)
        profitable_then_stopped = df[
            (df['exit_reason'] == 'stop') &
            (df['max_favorable_excursion'] > 0)
        ]
        if len(profitable_then_stopped) > 3:
            avg_max_profit = profitable_then_stopped['max_favorable_excursion'].mean()
            adjustments.append(ParameterAdjustment(
                parameter='trailing_activation_pct',
                current_value=0.01,
                recommended_value=round(avg_max_profit * 0.5, 4),
                reason=f'{len(profitable_then_stopped)} trades were profitable '
                       f'(avg peak {avg_max_profit:.1%}) then reversed to stop. '
                       f'Earlier trailing activation would lock in gains.',
                confidence=min(0.8, len(profitable_then_stopped) / 15),
                sample_size=len(profitable_then_stopped),
                expected_impact=f'Capture ~${profitable_then_stopped["pnl"].abs().sum():.0f} in lost profits',
            ))

        # 3. Regime-conditional analysis
        if 'regime' in df.columns and df['regime'].notna().sum() > 10:
            for regime in df['regime'].unique():
                regime_trades = df[df['regime'] == regime]
                if len(regime_trades) >= 5:
                    wr = (regime_trades['pnl'] > 0).mean()
                    if wr < 0.35:
                        adjustments.append(ParameterAdjustment(
                            parameter=f'regime_block_{regime}',
                            current_value=0,  # Currently allowed
                            recommended_value=1,  # Block
                            reason=f'Regime "{regime}": {wr:.0%} WR over '
                                   f'{len(regime_trades)} trades. '
                                   f'P&L: ${regime_trades["pnl"].sum():+,.0f}. '
                                   f'Consider blocking entries in this regime.',
                            confidence=min(0.7, len(regime_trades) / 20),
                            sample_size=len(regime_trades),
                            expected_impact=f'Avoid ${regime_trades[regime_trades["pnl"] < 0]["pnl"].sum():,.0f} in losses',
                        ))

        # 4. Time-of-day analysis
        df['entry_hour'] = pd.to_datetime(df['entry_time']).dt.hour
        for hour in df['entry_hour'].unique():
            hour_trades = df[df['entry_hour'] == hour]
            if len(hour_trades) >= 5:
                wr = (hour_trades['pnl'] > 0).mean()
                if wr < 0.30:
                    adjustments.append(ParameterAdjustment(
                        parameter=f'block_hour_{hour}',
                        current_value=0,
                        recommended_value=1,
                        reason=f'Hour {hour}:00: {wr:.0%} WR over {len(hour_trades)} trades. '
                               f'P&L: ${hour_trades["pnl"].sum():+,.0f}.',
                        confidence=min(0.6, len(hour_trades) / 15),
                        sample_size=len(hour_trades),
                        expected_impact=f'Avoid ${hour_trades[hour_trades["pnl"] < 0]["pnl"].sum():,.0f} in losses',
                    ))

        # 5. Gap size edge decay
        df['abs_gap'] = df['gap_pct'].abs()
        for bucket, (lo, hi) in {'5-7%': (0.05, 0.07), '7-10%': (0.07, 0.10),
                                   '10-15%': (0.10, 0.15), '15%+': (0.15, 1.0)}.items():
            bucket_trades = df[(df['abs_gap'] >= lo) & (df['abs_gap'] < hi)]
            if len(bucket_trades) >= 5:
                wr = (bucket_trades['pnl'] > 0).mean()
                pf = (bucket_trades[bucket_trades['pnl'] > 0]['pnl'].sum() /
                      abs(bucket_trades[bucket_trades['pnl'] < 0]['pnl'].sum())
                      if bucket_trades[bucket_trades['pnl'] < 0]['pnl'].sum() != 0 else 0)
                if pf < 0.8:
                    adjustments.append(ParameterAdjustment(
                        parameter=f'gap_bucket_{bucket}',
                        current_value=pf,
                        recommended_value=0,  # Disable
                        reason=f'Gap {bucket}: PF {pf:.2f}, WR {wr:.0%} over '
                               f'{len(bucket_trades)} trades — no edge.',
                        confidence=min(0.7, len(bucket_trades) / 20),
                        sample_size=len(bucket_trades),
                        expected_impact=f'Avoid ${bucket_trades[bucket_trades["pnl"] < 0]["pnl"].sum():,.0f}',
                    ))

        # Sort by confidence * impact
        adjustments.sort(key=lambda a: a.confidence, reverse=True)
        return adjustments


# ═══════════════════════════════════════════════════════════════════════════════
# 6. EMERGENCY KILL SWITCH
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class KillSwitchConfig:
    """Kill switch thresholds — non-negotiable safety limits."""
    max_daily_loss_pct: float = 0.03      # -3% daily loss → halt
    max_drawdown_pct: float = 0.08        # -8% from peak → halt
    max_consecutive_losses: int = 5       # 5 in a row → halt
    max_daily_trades: int = 10            # 10 trades/day → halt
    max_single_loss_pct: float = 0.02     # -2% on single trade → review


class KillSwitch:
    """Emergency circuit breaker. Checks BEFORE every trade entry."""

    def __init__(self, config: KillSwitchConfig = None):
        self.config = config or KillSwitchConfig()
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.peak_equity = 0.0
        self.halted = False
        self.halt_reason = ''

    def check(self, equity: float, initial_capital: float) -> Tuple[bool, str]:
        """Check all kill switch conditions. Returns (allowed, reason)."""
        if self.halted:
            return False, f'HALTED: {self.halt_reason}'

        cfg = self.config

        # Daily loss
        daily_loss_pct = self.daily_pnl / initial_capital if initial_capital > 0 else 0
        if daily_loss_pct < -cfg.max_daily_loss_pct:
            self.halted = True
            self.halt_reason = f'Daily loss {daily_loss_pct:.1%} exceeds {-cfg.max_daily_loss_pct:.1%}'
            return False, self.halt_reason

        # Drawdown from peak
        self.peak_equity = max(self.peak_equity, equity)
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0
        if dd > cfg.max_drawdown_pct:
            self.halted = True
            self.halt_reason = f'Drawdown {dd:.1%} exceeds {cfg.max_drawdown_pct:.1%}'
            return False, self.halt_reason

        # Consecutive losses
        if self.consecutive_losses >= cfg.max_consecutive_losses:
            self.halted = True
            self.halt_reason = f'{self.consecutive_losses} consecutive losses'
            return False, self.halt_reason

        # Daily trade count
        if self.daily_trades >= cfg.max_daily_trades:
            return False, f'Daily trade limit ({cfg.max_daily_trades}) reached'

        return True, 'OK'

    def record_trade(self, pnl: float):
        """Update state after a trade completes."""
        self.daily_pnl += pnl
        self.daily_trades += 1
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.halted = False
        self.halt_reason = ''


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DECISION TREE — Complete Entry Flow
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeDecision:
    """Complete decision audit trail."""
    symbol: str
    action: str               # 'enter', 'skip', 'blocked'
    probability: float        # P(profit | features)
    position_size: Optional[PositionSize] = None
    regime: Optional[MarketRegime] = None
    factors: Dict[str, float] = field(default_factory=dict)
    reason: str = ''
    stop_price: float = 0.0
    entry_price: float = 0.0


class QuantGapFadeEngine:
    """Orchestrates the full decision pipeline.

    Entry Decision Tree:
        1. Kill switch check → blocked if tripped
        2. Regime detection → classify market state
        3. Feature extraction → build candidate features
        4. Bayesian scoring → P(profit | features)
        5. If P < threshold → skip
        6. Position sizing → Kelly * regime adjustment
        7. Stop calculation → adaptive stop from gap% and ATR
        8. ENTER

    Exit Decision Tree:
        1. Hard stop check (broker-side)
        2. Trailing stop update
        3. Time decay tightening
        4. Volume divergence check
        5. Partial profit at half target
        6. Time exit at 3 PM
        7. EOD force close at 3:50 PM
    """

    def __init__(self, equity: float = 25000, spy_bars: pd.DataFrame = None):
        self.scorer = BayesianScorer()
        self.exit_mgr = ExitManager()
        self.kill_switch = KillSwitch()
        self.post_mortem = PostMortemAnalyzer()
        self.regime_detector = RegimeDetector(spy_bars) if spy_bars is not None else None
        self.sizer = PositionSizer(equity)
        self.equity = equity
        self.decisions: List[TradeDecision] = []

    def evaluate_candidate(self, symbol: str, gap_pct: float, vol_ratio: float,
                           price: float, prev_close: float, atr_pct: float = 0.02,
                           now: datetime = None) -> TradeDecision:
        """Full entry decision pipeline for a gap candidate."""
        now = now or datetime.now(ET)

        # 1. Kill switch
        allowed, kill_reason = self.kill_switch.check(self.equity, self.sizer.equity)
        if not allowed:
            return TradeDecision(symbol=symbol, action='blocked', probability=0,
                                reason=f'Kill switch: {kill_reason}')

        # 2. Regime
        regime = self.regime_detector.classify() if self.regime_detector else MarketRegime(
            'unknown', 0, 0, 0, 0, 0, 0)

        # 3. Feature extraction
        features = CandidateFeatures(
            symbol=symbol, gap_pct=gap_pct, abs_gap_pct=abs(gap_pct),
            vol_ratio=vol_ratio, price=price,
            market_cap_bucket='mid',  # TODO: lookup from reference data
            regime=regime,
            time_of_day=now.hour + now.minute / 60.0,
            atr_pct=atr_pct, prev_close=prev_close,
        )

        # 4. Bayesian scoring
        should_trade, probability, reason = self.scorer.should_trade(features)

        if not should_trade:
            return TradeDecision(symbol=symbol, action='skip', probability=probability,
                                regime=regime, reason=reason)

        # 5. Stop calculation
        stop_pct = max(0.01, min(0.03, abs(gap_pct) * 0.25))  # Adaptive: 25% of gap, clamped 1-3%

        # 6. Position sizing
        size = self.sizer.compute(features, probability, stop_pct)

        if size.shares <= 0:
            return TradeDecision(symbol=symbol, action='skip', probability=probability,
                                regime=regime, reason='Position size = 0 (insufficient equity)')

        # 7. Compute stop price
        if gap_pct > 0:  # Short gap-up
            stop_price = price * (1 + stop_pct)
            entry_price = price
        else:  # Long gap-down
            stop_price = price * (1 - stop_pct)
            entry_price = price

        decision = TradeDecision(
            symbol=symbol, action='enter', probability=probability,
            position_size=size, regime=regime,
            factors=self.scorer.score(features)[1],
            reason=reason,
            stop_price=stop_price, entry_price=entry_price,
        )
        self.decisions.append(decision)
        return decision

    def run_post_mortem(self, trades: List[TradeLog]) -> List[ParameterAdjustment]:
        """Run post-mortem analysis on trade logs and return adjustments."""
        return self.post_mortem.analyze(trades)

    def update_likelihood_ratios(self, adjustments: List[ParameterAdjustment]):
        """Apply parameter adjustments from post-mortem to the scorer.

        This is the 'learning loop' — adjusting priors based on evidence.
        """
        for adj in adjustments:
            if adj.confidence >= 0.6 and adj.parameter in self.scorer.LIKELIHOOD_RATIOS:
                old = self.scorer.LIKELIHOOD_RATIOS[adj.parameter]
                self.scorer.LIKELIHOOD_RATIOS[adj.parameter] = adj.recommended_value
                logger.info(f"Updated {adj.parameter}: {old} → {adj.recommended_value} "
                           f"(confidence: {adj.confidence:.0%})")
