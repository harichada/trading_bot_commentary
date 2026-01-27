"""
Human Trader Filter - Makes the bot trade like a sensible human

This module integrates all the sophisticated analysis components:
- Session timing (don't trade during lunch, be careful at open)
- Market regime (don't force trades in choppy markets)
- Recent P&L adjustment (scale down after losses)
- Psychological levels (round numbers matter)
- Risk scaling based on conviction and recent performance

A real trader doesn't just see a signal and hit buy. They consider:
1. "What time is it? Is this a good time to trade?"
2. "How is the market behaving? Trending or chopping?"
3. "How am I doing today? Should I be aggressive or defensive?"
4. "Are there any big round numbers nearby that could cause resistance?"
5. "Does this setup have multiple confirmations?"
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TradeDecision(Enum):
    """Decision outcomes"""
    TAKE_TRADE = "take_trade"
    REDUCE_SIZE = "reduce_size"
    SKIP_TRADE = "skip_trade"
    WAIT_FOR_BETTER_ENTRY = "wait_for_better_entry"


@dataclass
class FilterResult:
    """Result of human trader filter analysis"""
    decision: TradeDecision
    original_size_multiplier: float
    adjusted_size_multiplier: float
    confidence_adjustment: float
    reasons: List[str]
    warnings: List[str]
    session_status: str
    regime_status: str
    pnl_status: str
    psychological_levels: List[float]
    final_recommendation: str


@dataclass
class RecentPerformance:
    """Track recent trading performance for adaptive sizing"""
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    pnl_today: float = 0.0
    pnl_this_week: float = 0.0
    largest_loss_today: float = 0.0
    win_rate_7d: float = 0.5
    last_trade_result: Optional[str] = None  # 'win' or 'loss'


class HumanTraderFilter:
    """
    Filters trading signals the way a human trader would think.

    Core Philosophy:
    - Protect capital first, make money second
    - Don't trade when conditions are unfavorable
    - Scale position size based on conviction AND recent performance
    - Respect time of day and market regime
    - Be aware of psychological price levels
    """

    def __init__(self, session_manager=None, regime_detector=None, config: Optional[Dict] = None):
        self.session_manager = session_manager
        self.regime_detector = regime_detector
        self.config = config or {}

        # Recent performance tracking
        self.performance = RecentPerformance()
        self.trade_history: List[Dict] = []

        # Psychological level intervals (humans watch these)
        self.psych_levels = [1, 5, 10, 25, 50, 100, 250, 500, 1000]

        # Adaptive risk parameters
        self.base_risk_per_trade = self.config.get('base_risk_pct', 0.01)  # 1%
        self.max_risk_per_trade = self.config.get('max_risk_pct', 0.02)   # 2%
        self.min_risk_per_trade = self.config.get('min_risk_pct', 0.005) # 0.5%

        # Loss scaling (reduce after losses)
        self.loss_scale_factor = self.config.get('loss_scale_factor', 0.75)  # Reduce 25% per loss
        self.win_scale_factor = self.config.get('win_scale_factor', 1.1)     # Increase 10% per win
        self.max_consecutive_scale = 3  # Don't compound more than 3x

        # Session restrictions
        self.blocked_sessions = ['market_open', 'lunch_dead_zone', 'after_hours', 'pre_market']
        self.reduced_sessions = ['afternoon']  # Half size
        self.prime_sessions = ['morning_momentum', 'power_hour']  # Full size or more

        # Regime restrictions
        self.blocked_regimes = ['choppy']
        self.reduced_regimes = ['volatile', 'unknown']
        self.favorable_regimes = ['trending_up', 'trending_down', 'ranging']

        logger.info("HumanTraderFilter initialized - Trading like a human now")

    def evaluate_signal(
        self,
        symbol: str,
        signal_type: str,  # 'BUY' or 'SELL'
        signal_strength: float,
        current_price: float,
        market_data: Optional[pd.DataFrame] = None,
        additional_context: Optional[Dict] = None
    ) -> FilterResult:
        """
        Evaluate a trading signal the way a human trader would.

        Returns a FilterResult with the decision and reasoning.
        """
        reasons = []
        warnings = []
        size_multiplier = 1.0
        confidence_adj = 0.0

        # 1. CHECK TIME OF DAY (Session)
        session_status, session_mult, session_reasons = self._check_session()
        size_multiplier *= session_mult
        reasons.extend(session_reasons)

        if session_mult == 0:
            return FilterResult(
                decision=TradeDecision.SKIP_TRADE,
                original_size_multiplier=1.0,
                adjusted_size_multiplier=0.0,
                confidence_adjustment=-1.0,
                reasons=reasons,
                warnings=["Trading blocked during this session"],
                session_status=session_status,
                regime_status="not_checked",
                pnl_status="not_checked",
                psychological_levels=[],
                final_recommendation=f"SKIP: {session_status} - not a good time to trade"
            )

        # 2. CHECK MARKET REGIME
        regime_status, regime_mult, regime_reasons = self._check_regime(market_data)
        size_multiplier *= regime_mult
        reasons.extend(regime_reasons)

        if regime_mult == 0:
            return FilterResult(
                decision=TradeDecision.SKIP_TRADE,
                original_size_multiplier=1.0,
                adjusted_size_multiplier=0.0,
                confidence_adjustment=-1.0,
                reasons=reasons,
                warnings=["Market regime unfavorable for trading"],
                session_status=session_status,
                regime_status=regime_status,
                pnl_status="not_checked",
                psychological_levels=[],
                final_recommendation=f"SKIP: Market is {regime_status} - wait for better conditions"
            )

        # 3. CHECK RECENT P&L (Adaptive Risk)
        pnl_status, pnl_mult, pnl_reasons = self._check_recent_pnl()
        size_multiplier *= pnl_mult
        reasons.extend(pnl_reasons)

        # 4. CHECK PSYCHOLOGICAL LEVELS
        psych_levels = self._find_psychological_levels(current_price)
        psych_warnings = self._check_psych_level_proximity(current_price, psych_levels, signal_type)
        warnings.extend(psych_warnings)

        # 5. APPLY SIGNAL STRENGTH
        # Weak signals get reduced size
        if signal_strength < 0.6:
            size_multiplier *= 0.5
            reasons.append(f"Signal strength {signal_strength:.0%} is weak - half size")
        elif signal_strength > 0.85:
            size_multiplier *= 1.2
            reasons.append(f"Signal strength {signal_strength:.0%} is strong - increased size")

        # 6. FINAL DECISION
        final_multiplier = max(self.min_risk_per_trade / self.base_risk_per_trade,
                               min(size_multiplier, self.max_risk_per_trade / self.base_risk_per_trade))

        if final_multiplier < 0.25:
            decision = TradeDecision.SKIP_TRADE
            recommendation = "SKIP: Too many negative factors"
        elif final_multiplier < 0.75:
            decision = TradeDecision.REDUCE_SIZE
            recommendation = f"TAKE with {final_multiplier:.0%} size - proceed with caution"
        else:
            decision = TradeDecision.TAKE_TRADE
            recommendation = f"TAKE with {final_multiplier:.0%} size - conditions favorable"

        return FilterResult(
            decision=decision,
            original_size_multiplier=1.0,
            adjusted_size_multiplier=final_multiplier,
            confidence_adjustment=confidence_adj,
            reasons=reasons,
            warnings=warnings,
            session_status=session_status,
            regime_status=regime_status,
            pnl_status=pnl_status,
            psychological_levels=psych_levels,
            final_recommendation=recommendation
        )

    def _check_session(self) -> Tuple[str, float, List[str]]:
        """Check current market session and return multiplier"""
        reasons = []

        if not self.session_manager:
            return "unknown", 1.0, ["Session manager not available"]

        try:
            session_state = self.session_manager.get_current_session()
            session_name = session_state.session.value if hasattr(session_state.session, 'value') else str(session_state.session)

            if session_name in self.blocked_sessions:
                reasons.append(f"🚫 {session_name}: Not trading during this session")
                return session_name, 0.0, reasons

            if session_name in self.reduced_sessions:
                reasons.append(f"⚠️ {session_name}: Reducing position size")
                return session_name, 0.5, reasons

            if session_name in self.prime_sessions:
                reasons.append(f"✅ {session_name}: Prime trading time")
                return session_name, 1.2, reasons

            return session_name, 1.0, reasons

        except Exception as e:
            logger.warning(f"Session check failed: {e}")
            return "error", 1.0, [f"Session check error: {e}"]

    def _check_regime(self, market_data: Optional[pd.DataFrame]) -> Tuple[str, float, List[str]]:
        """Check market regime and return multiplier"""
        reasons = []

        if not self.regime_detector:
            return "unknown", 1.0, ["Regime detector not available"]

        try:
            if market_data is not None and len(market_data) > 20:
                regime_state = self.regime_detector.detect_regime(market_data)
            else:
                regime_state = self.regime_detector.get_current_state()

            regime_name = regime_state.regime.value if hasattr(regime_state.regime, 'value') else str(regime_state.regime)

            if regime_name in self.blocked_regimes:
                reasons.append(f"🚫 Market is {regime_name}: Don't trade choppy markets")
                return regime_name, 0.0, reasons

            if regime_name in self.reduced_regimes:
                reasons.append(f"⚠️ Market is {regime_name}: Reduce position size")
                return regime_name, 0.5, reasons

            if regime_name in self.favorable_regimes:
                reasons.append(f"✅ Market is {regime_name}: Favorable conditions")
                # Extra boost for strong trends
                if hasattr(regime_state, 'trend_strength') and regime_state.trend_strength > 40:
                    reasons.append(f"Strong trend (ADX={regime_state.trend_strength:.0f}): Increased conviction")
                    return regime_name, 1.2, reasons
                return regime_name, 1.0, reasons

            return regime_name, 0.8, reasons

        except Exception as e:
            logger.warning(f"Regime check failed: {e}")
            return "error", 1.0, [f"Regime check error: {e}"]

    def _check_recent_pnl(self) -> Tuple[str, float, List[str]]:
        """Check recent P&L and adjust risk accordingly"""
        reasons = []
        multiplier = 1.0

        # Scale down after consecutive losses
        if self.performance.consecutive_losses > 0:
            loss_penalty = self.loss_scale_factor ** min(self.performance.consecutive_losses, self.max_consecutive_scale)
            multiplier *= loss_penalty
            reasons.append(f"⚠️ {self.performance.consecutive_losses} consecutive losses: Size reduced to {loss_penalty:.0%}")

        # Scale up after consecutive wins (carefully)
        elif self.performance.consecutive_wins >= 2:
            win_bonus = self.win_scale_factor ** min(self.performance.consecutive_wins - 1, self.max_consecutive_scale)
            multiplier *= min(win_bonus, 1.5)  # Cap at 1.5x
            reasons.append(f"✅ {self.performance.consecutive_wins} consecutive wins: Size increased to {min(win_bonus, 1.5):.0%}")

        # Daily loss limit check
        daily_pnl_pct = self.performance.pnl_today / 10000  # Assume $10k account for %
        if daily_pnl_pct < -0.02:  # Down more than 2%
            multiplier *= 0.25
            reasons.append(f"🛑 Down {abs(daily_pnl_pct):.1%} today: Heavily reduced size (capital preservation)")
        elif daily_pnl_pct < -0.01:  # Down more than 1%
            multiplier *= 0.5
            reasons.append(f"⚠️ Down {abs(daily_pnl_pct):.1%} today: Reduced size")

        # Max trades per day check
        if self.performance.trades_today >= 10:
            reasons.append("🛑 Max trades for today reached")
            return "max_trades", 0.0, reasons

        status = f"Today: {self.performance.wins_today}W/{self.performance.losses_today}L, ${self.performance.pnl_today:+.2f}"
        return status, multiplier, reasons

    def _find_psychological_levels(self, price: float) -> List[float]:
        """Find nearby psychological price levels"""
        levels = []

        for interval in self.psych_levels:
            if interval > price * 2:  # Don't look at levels way above price
                break

            # Find nearest levels above and below
            level_below = (price // interval) * interval
            level_above = level_below + interval

            # Only include if within 5% of price
            if abs(level_below - price) / price < 0.05:
                levels.append(level_below)
            if abs(level_above - price) / price < 0.05:
                levels.append(level_above)

        return sorted(set(levels))

    def _check_psych_level_proximity(self, price: float, levels: List[float], signal_type: str) -> List[str]:
        """Check if psychological levels might cause issues"""
        warnings = []

        for level in levels:
            distance_pct = (level - price) / price * 100

            if signal_type == 'BUY':
                # Warning if resistance is close above
                if 0 < distance_pct < 2:
                    warnings.append(f"⚠️ Psychological resistance at ${level:.2f} ({distance_pct:.1f}% above)")
            else:  # SELL
                # Warning if support is close below
                if -2 < distance_pct < 0:
                    warnings.append(f"⚠️ Psychological support at ${level:.2f} ({abs(distance_pct):.1f}% below)")

        return warnings

    def record_trade_result(self, symbol: str, pnl: float, is_win: bool):
        """Record trade result for adaptive sizing"""
        self.performance.trades_today += 1
        self.performance.pnl_today += pnl

        if is_win:
            self.performance.wins_today += 1
            self.performance.consecutive_wins += 1
            self.performance.consecutive_losses = 0
            self.performance.last_trade_result = 'win'
        else:
            self.performance.losses_today += 1
            self.performance.consecutive_losses += 1
            self.performance.consecutive_wins = 0
            self.performance.last_trade_result = 'loss'
            if pnl < self.performance.largest_loss_today:
                self.performance.largest_loss_today = pnl

        self.trade_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'pnl': pnl,
            'is_win': is_win
        })

        logger.info(f"Trade recorded: {symbol} {'WIN' if is_win else 'LOSS'} ${pnl:.2f} | "
                   f"Streak: {self.performance.consecutive_wins}W/{self.performance.consecutive_losses}L")

    def reset_daily_stats(self):
        """Reset daily stats (call at market open)"""
        self.performance.trades_today = 0
        self.performance.wins_today = 0
        self.performance.losses_today = 0
        self.performance.pnl_today = 0.0
        self.performance.largest_loss_today = 0.0
        # Don't reset consecutive counts - they carry over
        logger.info("Daily stats reset")

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current filter status for display"""
        session_status = "unknown"
        regime_status = "unknown"

        if self.session_manager:
            try:
                session = self.session_manager.get_current_session()
                session_status = session.session.value if hasattr(session.session, 'value') else str(session.session)
            except:
                pass

        if self.regime_detector:
            try:
                regime = self.regime_detector.get_current_state()
                regime_status = regime.regime.value if hasattr(regime.regime, 'value') else str(regime.regime)
            except:
                pass

        return {
            'session': session_status,
            'regime': regime_status,
            'trades_today': self.performance.trades_today,
            'pnl_today': self.performance.pnl_today,
            'win_streak': self.performance.consecutive_wins,
            'loss_streak': self.performance.consecutive_losses,
            'win_rate_today': (self.performance.wins_today / max(1, self.performance.trades_today)) * 100
        }
