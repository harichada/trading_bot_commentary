"""Minervini Trend Template Strategy — Strategy #4 (Stage 2 Trend Following).

Implements Mark Minervini's 8-criterion Trend Template to identify Stage 2
uptrend stocks.  Dual-purpose filter for the gap fade bot:

- **Gap-down longs** (direction='long'): If a stock gaps down but is in Stage 2
  uptrend, this is a high-probability pullback entry → BUY.
- **Gap-up shorts** (direction='short'): If a stock is in Stage 2, don't short
  it — the gap-up is likely continuation, not exhaustion → REJECT.

The 8 Minervini criteria:
1. Price > 150 SMA
2. Price > 200 SMA
3. 150 SMA > 200 SMA
4. 200 SMA rising over past 22 trading days
5. 50 SMA > 150 SMA AND 50 SMA > 200 SMA
6. Price > 50 SMA
7. Price ≥ 25% above 52-week low
8. Price within 25% of 52-week high
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .base import ExitSignal, GapFadeStrategy
from .registry import GapFadeStrategyRegistry

logger = logging.getLogger(__name__)

# ── Default config ─────────────────────────────────────────────────────

_DEFAULTS = {
    'lookback_days': 252,
    'sma_50_period': 50,
    'sma_150_period': 150,
    'sma_200_period': 200,
    'sma_200_uptrend_days': 22,
    'pct_above_52w_low': 0.25,
    'pct_within_52w_high': 0.25,
    'min_criteria': 6,
    'min_confidence': 0.60,
    'reject_stage2_shorts': True,
    'score_boost_max': 40.0,
    'score_boost_min': 10.0,
    'entry_after_minute': 35,
    'entry_cutoff_hour': 11,
    'entry_cutoff_minute': 0,
    'require_volume_confirmation': True,
    'volume_confirm_ratio': 1.0,
    # Stops & targets: handled by engine (GapFadeConfig.stop_pct)
    'stage_deterioration_exit': True,
    'profit_drawdown_pct': 0.40,
    'profit_protect_trigger_pct': 0.05,
}


# ── Data structures ────────────────────────────────────────────────────

@dataclass
class MinerviniCriteria:
    """The 8 Minervini Trend Template criteria."""
    price_above_150sma: bool = False
    price_above_200sma: bool = False
    sma150_above_200sma: bool = False
    sma200_trending_up: bool = False
    sma50_above_150_and_200: bool = False
    price_above_50sma: bool = False
    pct_above_52w_low: bool = False
    within_25pct_52w_high: bool = False

    @property
    def count(self) -> int:
        return sum([
            self.price_above_150sma,
            self.price_above_200sma,
            self.sma150_above_200sma,
            self.sma200_trending_up,
            self.sma50_above_150_and_200,
            self.price_above_50sma,
            self.pct_above_52w_low,
            self.within_25pct_52w_high,
        ])


@dataclass
class MinerviniAnalysis:
    """Cached per-(symbol, date) analysis result."""
    symbol: str
    computed_date: str
    criteria: MinerviniCriteria
    stage: int                  # 1-4
    confidence: float           # 0.0-1.0
    relative_strength: float    # ratio vs SPY, >1.0 = outperforming
    sma_50: float
    sma_150: float
    sma_200: float
    current_price: float
    low_52w: float
    high_52w: float
    low_50d: float
    atr_14: float


# ── Strategy implementation ────────────────────────────────────────────

@GapFadeStrategyRegistry.register('minervini_trend')
class MinerviniTrendStrategy(GapFadeStrategy):
    """Minervini Trend Template: 8-criteria Stage 2 trend filter.

    Longs: buy pullbacks in Stage 2 uptrend stocks (gap-down entries).
    Shorts: reject Stage 2 stocks (don't short strong uptrends).
    """

    name = 'Minervini Trend Template'
    description = (
        'Mark Minervini 8-criterion Stage 2 trend filter. Buys gap-down '
        'pullbacks in strong uptrends, rejects shorting stocks in Stage 2. '
        'Stops at 50-day low, targets at 1.5R/3R multiples.'
    )
    version = '1.0'

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self._cache: Dict[str, MinerviniAnalysis] = {}
        self._spy_cache: Dict[str, object] = {}  # date_range → SPY DataFrame
        self._insufficient: set = set()  # symbols with <200 bars (skip fast)

    def get_default_config(self) -> Dict:
        return dict(_DEFAULTS)

    # ── Technical analysis helpers ─────────────────────────────────────

    def _get_spy_cached(self, db, start: str, end: str):
        """Fetch SPY bars with caching across symbols for the same date range."""
        cache_key = f'{start}_{end}'
        if cache_key in self._spy_cache:
            return self._spy_cache[cache_key]
        spy_df = db.get_bars('SPY', start, end)
        self._spy_cache[cache_key] = spy_df
        return spy_df

    def _compute_analysis(self, symbol: str, as_of_date: str) -> Optional[MinerviniAnalysis]:
        """Compute or return cached MinerviniAnalysis for symbol."""
        cache_key = f'{symbol}_{as_of_date}'
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # Fast skip for symbols we already know have insufficient data
        if symbol in self._insufficient:
            return None

        from gap_fade_app import get_price_db
        db = get_price_db()

        lookback = self._config.get('lookback_days', 252)
        try:
            from datetime import date
            end_dt = date.fromisoformat(as_of_date)
            start_dt = end_dt - timedelta(days=int(lookback * 1.5))
            start = start_dt.isoformat()
        except (ValueError, TypeError):
            logger.warning(f"minervini_trend: invalid as_of_date={as_of_date}")
            return None

        df = db.get_bars(symbol, start, as_of_date)
        if df is None or len(df) < 200:
            self._insufficient.add(symbol)
            logger.debug(
                f"minervini_trend: {symbol} only "
                f"{len(df) if df is not None else 0} bars, need 200+"
            )
            return None

        # Load SPY for relative strength (cached across symbols)
        spy_df = self._get_spy_cached(db, start, as_of_date)

        import numpy as _np

        closes = df['close'].values
        n = len(closes)

        # SMAs — use numpy cumsum for O(n) instead of pandas rolling
        sma_50_period = self._config.get('sma_50_period', 50)
        sma_150_period = self._config.get('sma_150_period', 150)
        sma_200_period = self._config.get('sma_200_period', 200)

        def _sma_last(arr, period):
            """Fast SMA of last `period` values."""
            if len(arr) < period:
                return float('nan')
            return float(arr[-period:].mean())

        sma_50 = _sma_last(closes, sma_50_period)
        sma_150 = _sma_last(closes, sma_150_period)
        sma_200 = _sma_last(closes, sma_200_period)

        current_price = float(closes[-1])

        # 52-week high/low (use all available bars, up to 252)
        lookback_52w = min(252, n)
        high_52w = float(df['high'].values[-lookback_52w:].max())
        low_52w = float(df['low'].values[-lookback_52w:].min())

        # 50-day low
        lookback_50d = min(50, n)
        low_50d = float(df['low'].values[-lookback_50d:].min())

        # ATR-14
        atr_14 = self._compute_atr(df, period=14)

        # ── Evaluate 8 criteria ──
        uptrend_days = self._config.get('sma_200_uptrend_days', 22)
        pct_above_low = self._config.get('pct_above_52w_low', 0.25)
        pct_within_high = self._config.get('pct_within_52w_high', 0.25)

        # Criterion 4: 200 SMA rising — compare current vs N days ago
        sma200_rising = False
        if n > sma_200_period + uptrend_days:
            sma200_past = _sma_last(closes[:-(uptrend_days)], sma_200_period)
            if sma_200 > sma200_past:
                sma200_rising = True

        criteria = MinerviniCriteria(
            price_above_150sma=current_price > sma_150,
            price_above_200sma=current_price > sma_200,
            sma150_above_200sma=sma_150 > sma_200,
            sma200_trending_up=sma200_rising,
            sma50_above_150_and_200=(sma_50 > sma_150 and sma_50 > sma_200),
            price_above_50sma=current_price > sma_50,
            pct_above_52w_low=(
                low_52w > 0 and
                (current_price - low_52w) / low_52w >= pct_above_low
            ),
            within_25pct_52w_high=(
                high_52w > 0 and
                (high_52w - current_price) / high_52w <= pct_within_high
            ),
        )

        # Stage classification
        stage = self._classify_stage(criteria, sma_50, sma_150, sma_200, current_price)

        # Confidence = criteria_met / 8
        confidence = criteria.count / 8.0

        # Relative strength vs SPY
        rs = self._compute_relative_strength(df, spy_df, lookback=63)

        analysis = MinerviniAnalysis(
            symbol=symbol,
            computed_date=as_of_date,
            criteria=criteria,
            stage=stage,
            confidence=confidence,
            relative_strength=rs,
            sma_50=sma_50,
            sma_150=sma_150,
            sma_200=sma_200,
            current_price=current_price,
            low_52w=low_52w,
            high_52w=high_52w,
            low_50d=low_50d,
            atr_14=atr_14,
        )
        self._cache[cache_key] = analysis
        # Also cache under just symbol for stop/target/exit lookups
        self._cache[symbol] = analysis
        return analysis

    @staticmethod
    def _compute_relative_strength(symbol_df, spy_df, lookback: int = 63) -> float:
        """Compute relative strength vs SPY over lookback period.

        Returns ratio: >1.0 means stock outperforms SPY.
        """
        if spy_df is None or len(spy_df) < lookback or len(symbol_df) < lookback:
            return 1.0  # neutral fallback

        sym_closes = symbol_df['close'].values
        spy_closes = spy_df['close'].values

        sym_return = (sym_closes[-1] / sym_closes[-lookback]) - 1.0
        spy_return = (spy_closes[-1] / spy_closes[-lookback]) - 1.0

        if abs(spy_return) < 1e-9:
            return 1.0

        # Ratio of returns (handle negative SPY)
        if spy_return > 0:
            return (1.0 + sym_return) / (1.0 + spy_return)
        else:
            # Both negative or mixed — compare magnitude
            return (1.0 + sym_return) / (1.0 + spy_return)

    @staticmethod
    def _classify_stage(criteria: MinerviniCriteria,
                        sma_50: float, sma_150: float, sma_200: float,
                        price: float) -> int:
        """Classify stock into Minervini Stage 1-4.

        Stage 2: ≥6 criteria AND price > sma_50 > sma_150 > sma_200
        Stage 4: price < sma_50 AND sma_50 < sma_150 (downtrend)
        Stage 3: 3-5 criteria, MA alignment broken (distribution)
        Stage 1: everything else (basing/accumulation)
        """
        full_alignment = (price > sma_50 > sma_150 > sma_200)

        if criteria.count >= 6 and full_alignment:
            return 2
        if price < sma_50 and sma_50 < sma_150:
            return 4
        if 3 <= criteria.count <= 5 and not full_alignment:
            return 3
        return 1

    @staticmethod
    def _compute_atr(df, period: int = 14) -> float:
        """Compute Average True Range."""
        if len(df) < period + 1:
            return 0.0

        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values

        true_ranges = []
        for i in range(1, len(df)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges) if true_ranges else 0.0

        return sum(true_ranges[-period:]) / period

    # ── Strategy hooks ─────────────────────────────────────────────────

    def filter_candidate(self, candidate: dict) -> Tuple[bool, str]:
        """Pre-filter based on Minervini Stage analysis.

        Longs: pass if Stage 2 with ≥min_criteria met.
        Shorts: reject if Stage 2 (don't short uptrends), pass otherwise.
        """
        symbol = candidate.get('symbol', '')
        if not symbol:
            return False, 'no symbol'

        as_of_date = candidate.get('date', '')
        if not as_of_date:
            as_of_date = datetime.now().strftime('%Y-%m-%d')

        analysis = self._compute_analysis(symbol, as_of_date)
        if analysis is None:
            # Insufficient data — let engine decide (pass through)
            return True, 'insufficient data for Minervini analysis, passing through'

        direction = candidate.get('direction', 'short')
        min_criteria = self._config.get('min_criteria', 6)
        reject_stage2_shorts = self._config.get('reject_stage2_shorts', True)

        if direction == 'long':
            # Gap-down long: require Stage 2 for pullback entry
            if analysis.stage == 2 and analysis.criteria.count >= min_criteria:
                return True, (
                    f'Stage 2 pullback: {analysis.criteria.count}/8 criteria, '
                    f'RS={analysis.relative_strength:.2f}, '
                    f'confidence={analysis.confidence:.2f}'
                )
            return False, (
                f'not Stage 2 (stage={analysis.stage}, '
                f'criteria={analysis.criteria.count}/8, need {min_criteria}+)'
            )
        else:
            # Gap-up short: reject if Stage 2 (don't short uptrends)
            if reject_stage2_shorts and analysis.stage == 2:
                return False, (
                    f'REJECT short: Stage 2 uptrend '
                    f'({analysis.criteria.count}/8 criteria, '
                    f'RS={analysis.relative_strength:.2f}) — '
                    f'gap-up likely continuation'
                )
            return True, (
                f'short allowed: stage={analysis.stage}, '
                f'criteria={analysis.criteria.count}/8'
            )

    def score_candidate(self, candidate: dict, regime: Optional[dict] = None) -> Optional[float]:
        """Boost score for long candidates based on criteria count + RS."""
        direction = candidate.get('direction', 'short')
        if direction != 'long':
            return None  # shorts use engine default

        symbol = candidate.get('symbol', '')
        analysis = self._cache.get(symbol)
        if analysis is None:
            return None

        base_score = candidate.get('score', 50.0)
        boost_min = self._config.get('score_boost_min', 10.0)
        boost_max = self._config.get('score_boost_max', 40.0)

        # Scale: 6/8 criteria → boost_min, 8/8 → boost_max
        criteria_range = max(analysis.criteria.count - 6, 0) / 2.0  # 0.0 to 1.0
        boost = boost_min + criteria_range * (boost_max - boost_min)

        # RS bonus: +5 per 0.1 above 1.0 (capped at +10)
        rs_bonus = min(max((analysis.relative_strength - 1.0) * 50.0, 0.0), 10.0)
        boost += rs_bonus

        boost = max(boost_min, min(boost_max, boost))
        return base_score + boost

    # ── Entry ──────────────────────────────────────────────────────────

    def get_entry_window(self) -> Optional[Tuple[int, int, int, int]]:
        """9:35 to 11:00 — slightly earlier than confluence (trend pullbacks)."""
        entry_min = self._config.get('entry_after_minute', 35)
        cutoff_h = self._config.get('entry_cutoff_hour', 11)
        cutoff_m = self._config.get('entry_cutoff_minute', 0)
        return (9, entry_min, cutoff_h, cutoff_m)

    def should_enter_now(self, candidate: dict, price: float,
                         tick_data: Optional[dict] = None,
                         now: Optional[datetime] = None) -> Tuple[bool, str]:
        """Gate long entries on volume confirmation. Shorts pass through."""
        direction = candidate.get('direction', 'short')
        if direction != 'long':
            return True, 'short passthrough'

        if not self._config.get('require_volume_confirmation', True):
            return True, 'volume confirmation disabled'

        if tick_data is None:
            return True, 'no tick data (backtest passthrough)'

        vol_ratio = tick_data.get('volume_ratio', None)
        if vol_ratio is None:
            return True, 'no volume data available'

        required = self._config.get('volume_confirm_ratio', 1.0)
        if vol_ratio >= required:
            return True, f'volume confirmed: ratio={vol_ratio:.2f} >= {required:.2f}'

        return False, f'waiting for volume: ratio={vol_ratio:.2f} < {required:.2f}'

    # ── Stop & Targets ─────────────────────────────────────────────────
    # Stops and targets are handled entirely by the engine (GapFadeConfig.stop_pct).
    # Returning None from both methods = "use engine defaults".

    # ── Exit Management ────────────────────────────────────────────────

    def evaluate_exit(self, position: dict, price: float, high: float,
                      tick_data: Optional[dict] = None,
                      now: Optional[datetime] = None) -> Optional[ExitSignal]:
        """Longs: exit on Stage 4 deterioration or profit drawdown. Shorts: None."""
        direction = position.get('direction', 'short')
        if direction != 'long':
            return None

        symbol = position.get('symbol', '')
        remaining = position.get('remaining_shares', 0)
        if remaining <= 0:
            return None

        entry_price = position.get('entry_price', 0)
        peak_price = position.get('peak_price', price)

        # ── Stage deterioration exit ──
        if self._config.get('stage_deterioration_exit', True):
            analysis = self._cache.get(symbol)
            if analysis and analysis.stage == 4:
                return ExitSignal(
                    action='close',
                    reason=(
                        f'stage_deterioration: {symbol} fell to Stage 4 '
                        f'(criteria={analysis.criteria.count}/8)'
                    ),
                    shares=remaining,
                )

        # ── Profit drawdown exit ──
        if entry_price > 0 and peak_price > entry_price:
            trigger_pct = self._config.get('profit_protect_trigger_pct', 0.05)
            drawdown_pct = self._config.get('profit_drawdown_pct', 0.40)

            profit_pct = (peak_price - entry_price) / entry_price
            if profit_pct >= trigger_pct:
                # Have meaningful profit — check if giving too much back
                current_profit = (price - entry_price) / entry_price
                given_back = profit_pct - current_profit
                if profit_pct > 0 and given_back / profit_pct >= drawdown_pct:
                    return ExitSignal(
                        action='close',
                        reason=(
                            f'profit_drawdown: gave back {given_back / profit_pct:.0%} '
                            f'of {profit_pct:.1%} peak profit '
                            f'(threshold={drawdown_pct:.0%})'
                        ),
                        shares=remaining,
                    )

        return None

    # Trailing stop: handled by engine (returns None = use engine default).

    # ── Indicators ─────────────────────────────────────────────────────

    def get_required_indicators(self) -> List[str]:
        return ['vwap', 'day_high', 'day_low']

    # ── LLM Prompt Overrides ───────────────────────────────────────────

    def get_llm_system_prompt(self) -> Optional[str]:
        return (
            "You are an autonomous trading supervisor for a Minervini Trend Template strategy.\n"
            "The bot uses Mark Minervini's 8-criterion Stage Analysis to identify stocks in\n"
            "Stage 2 (advancing) uptrends and trades gap-down pullbacks as long entries.\n\n"
            "The 8 Minervini Criteria (all must be checked):\n"
            "1. Price > 150-day SMA\n"
            "2. Price > 200-day SMA\n"
            "3. 150 SMA > 200 SMA\n"
            "4. 200 SMA trending up (rising over 22 trading days)\n"
            "5. 50 SMA > both 150 and 200 SMAs\n"
            "6. Price > 50 SMA\n"
            "7. Price >= 25% above 52-week low\n"
            "8. Price within 25% of 52-week high\n\n"
            "Stage Classification:\n"
            "- Stage 1: Basing/accumulation — MAs converging, price range-bound\n"
            "- Stage 2: Advancing — all MAs aligned, criteria met → BUY pullbacks\n"
            "- Stage 3: Distribution — MAs flattening, criteria failing\n"
            "- Stage 4: Declining — price below MAs, MAs declining → AVOID\n\n"
            "Strategy rules:\n"
            "- LONG gap-down pullbacks in Stage 2 stocks (6+/8 criteria met)\n"
            "- REJECT shorting Stage 2 stocks (gap-up = continuation, not exhaustion)\n"
            "- Stop loss and targets are managed by the engine (configurable stop_pct)\n"
            "- Exit on Stage 4 deterioration or 40% profit drawdown from peak\n\n"
            "Your responsibilities:\n"
            "1. CANDIDATE SELECTION: Only approve Stage 2 stocks with 6+ criteria.\n"
            "2. TREND VALIDATION: Verify MA alignment and relative strength vs SPY.\n"
            "3. RISK MANAGEMENT: Engine handles stops — focus on stage validation.\n"
            "4. EXIT DISCIPLINE: Close on stage deterioration — don't hold into Stage 4.\n"
            "5. After 11:00 AM ET, do NOT enter new positions.\n\n"
            "Respond ONLY with valid JSON. No text outside the JSON object."
        )

    def get_llm_profit_prompt(self) -> Optional[str]:
        return (
            "You are a profit-taking advisor for a Minervini Trend Template strategy.\n"
            "The bot buys gap-down pullbacks in Stage 2 uptrend stocks.\n\n"
            "You evaluate open LONG positions that are IN PROFIT and decide:\n"
            "- CLOSE: Take profit if Stage deteriorating or drawdown threshold hit.\n"
            "- HOLD: Let it run. DEFAULT bias. Trend stocks can run for weeks.\n"
            "- TIGHTEN_STOP: Raise stop to 50-day low as it rises.\n\n"
            "CRITICAL PRINCIPLES:\n"
            "- YOUR DEFAULT SHOULD BE HOLD. Trend-following means letting winners run.\n"
            "- If all 8 criteria still met and RS > 1.0: HOLD.\n"
            "- If criteria dropping (7→6→5): TIGHTEN_STOP.\n"
            "- If Stage drops to 3 or 4: CLOSE.\n"
            "- Stop loss and targets are engine-managed — focus on trend health.\n"
            "- If profit gives back 40% of peak: CLOSE.\n\n"
            "Respond ONLY with valid JSON. No text outside the JSON object."
        )

    # ── UI / Config ────────────────────────────────────────────────────

    def get_parameter_schema(self) -> Dict:
        return {
            'lookback_days': {
                'type': 'int', 'label': 'Lookback (trading days)',
                'default': 252, 'min': 200, 'max': 504,
                'description': 'Trading days of history for SMA and 52-week calcs',
            },
            'min_criteria': {
                'type': 'int', 'label': 'Min Criteria (of 8)',
                'default': 6, 'min': 4, 'max': 8,
                'description': 'Minimum Minervini criteria met for Stage 2 classification',
            },
            'min_confidence': {
                'type': 'float', 'label': 'Min Confidence',
                'default': 0.60, 'min': 0.40, 'max': 0.90,
                'description': 'Minimum confidence (criteria/8) to generate a signal',
            },
            'reject_stage2_shorts': {
                'type': 'bool', 'label': 'Reject Stage 2 Shorts',
                'default': True,
                'description': 'Block shorting stocks in Stage 2 uptrend',
            },
            'score_boost_min': {
                'type': 'float', 'label': 'Score Boost Min',
                'default': 10.0, 'min': 0.0, 'max': 30.0,
                'description': 'Minimum score boost for qualifying longs',
            },
            'score_boost_max': {
                'type': 'float', 'label': 'Score Boost Max',
                'default': 40.0, 'min': 10.0, 'max': 60.0,
                'description': 'Maximum score boost for 8/8 criteria + high RS',
            },
            'entry_after_minute': {
                'type': 'int', 'label': 'Entry After Minute',
                'default': 35, 'min': 31, 'max': 59,
                'description': 'Minute past 9:XX to start entries (e.g. 35 = 9:35 AM)',
            },
            'require_volume_confirmation': {
                'type': 'bool', 'label': 'Require Volume Confirmation',
                'default': True,
                'description': 'Gate long entries on volume ratio vs average',
            },
            'volume_confirm_ratio': {
                'type': 'float', 'label': 'Volume Confirm Ratio',
                'default': 1.0, 'min': 0.5, 'max': 3.0,
                'description': 'Volume ratio threshold for entry confirmation',
            },
            # Stops & targets: managed by engine (GapFadeConfig.stop_pct in main config)
            'stage_deterioration_exit': {
                'type': 'bool', 'label': 'Stage Deterioration Exit',
                'default': True,
                'description': 'Close position if stock drops to Stage 4',
            },
            'profit_drawdown_pct': {
                'type': 'float', 'label': 'Profit Drawdown %',
                'default': 0.40, 'min': 0.20, 'max': 0.60,
                'description': 'Close if this % of peak profit is given back',
            },
            'profit_protect_trigger_pct': {
                'type': 'float', 'label': 'Profit Protect Trigger %',
                'default': 0.05, 'min': 0.02, 'max': 0.15,
                'description': 'Profit % threshold before drawdown protection activates',
            },
        }
