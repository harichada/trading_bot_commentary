"""
No Trade Filter - Trade Quality Gate System
Blocks low-quality setups that retail traders lose money on

This is the MOST IMPORTANT module. The best trade is often NO TRADE.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, date, timedelta
import logging

# Import our modules
from regime_detector import get_regime_detector, MarketRegime
from session_manager import get_session_manager, MarketSession

logger = logging.getLogger(__name__)


class RejectReason(Enum):
    """Reasons for rejecting a trade"""
    CHOPPY_MARKET = "choppy_market"
    DEAD_ZONE = "dead_zone"
    DAILY_R_TARGET_HIT = "daily_r_target_hit"
    DRAWDOWN_MODE = "drawdown_mode"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    LOW_CONFLUENCE = "low_confluence"
    WEAK_SETUP = "weak_setup"
    NEWS_DAY = "news_day"
    OVERTRADING = "overtrading"
    CORRELATED_EXPOSURE = "correlated_exposure"
    REGIME_MISMATCH = "regime_mismatch"
    SESSION_BLOCKED = "session_blocked"
    VOLATILITY_TOO_HIGH = "volatility_too_high"
    VOLATILITY_TOO_LOW = "volatility_too_low"
    WIN_RATE_FILTER = "win_rate_filter"
    SIZE_LIMIT = "size_limit"


@dataclass
class TradeSignal:
    """Incoming trade signal to be filtered"""
    symbol: str
    direction: str  # 'long' or 'short'
    strategy_name: str
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_amount: float  # Dollar risk
    position_size: float  # Number of shares
    additional_signals: Dict[str, Any] = field(default_factory=dict)  # Supporting signals
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FilterResult:
    """Result of trade filtering"""
    allowed: bool
    signal: TradeSignal
    reject_reasons: List[RejectReason]
    warnings: List[str]
    adjusted_size: Optional[float]  # Reduced size if applicable
    confluence_score: float
    quality_score: float  # Overall setup quality 0-100
    message: str


@dataclass
class DailyStats:
    """Daily trading statistics for filtering decisions"""
    date: date
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0  # Total R-multiple P&L
    max_drawdown_r: float = 0.0
    consecutive_losses: int = 0
    is_in_drawdown: bool = False
    r_target_hit: bool = False


class NoTradeFilter:
    """
    The gatekeaper that protects you from yourself.

    Philosophy:
    - The best trade is NO trade (when conditions are wrong)
    - Quality over quantity - fewer trades, better results
    - Protect capital during drawdowns
    - Stop trading when daily R target is hit
    - Don't fight unfavorable conditions

    "The goal is not to trade more - it's to keep what you make."
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        # Get other modules
        self.regime_detector = get_regime_detector()
        self.session_manager = get_session_manager()

        # Daily R targets
        self.daily_r_target = self.config.get('daily_r_target', 3.0)  # Stop trading after +3R
        self.daily_r_loss_limit = self.config.get('daily_r_loss_limit', -2.0)  # Stop after -2R

        # Drawdown thresholds
        self.drawdown_threshold = self.config.get('drawdown_threshold', -1.5)  # R to trigger drawdown mode
        self.drawdown_size_reduction = self.config.get('drawdown_size_reduction', 0.5)  # 50% size in drawdown

        # Consecutive loss limits
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 3)
        self.loss_cooldown_minutes = self.config.get('loss_cooldown_minutes', 30)

        # Confluence requirements
        self.min_confluence_score = self.config.get('min_confluence_score', 2)  # At least 2 confirming signals
        self.min_quality_score = self.config.get('min_quality_score', 60)  # Out of 100

        # Trade limits
        self.max_trades_per_day = self.config.get('max_trades_per_day', 5)
        self.max_trades_per_symbol = self.config.get('max_trades_per_symbol', 2)

        # Confidence thresholds
        self.min_confidence = self.config.get('min_confidence', 0.70)
        self.min_r_multiple = self.config.get('min_r_multiple', 1.5)  # Risk:Reward

        # Strategy win rate filters
        self.min_strategy_win_rate = self.config.get('min_strategy_win_rate', 0.40)
        self.min_strategy_trades = self.config.get('min_strategy_trades', 10)  # Min trades for valid win rate

        # News/Events
        self.block_on_major_news = self.config.get('block_on_major_news', True)
        self.news_blackout_symbols: List[str] = []  # Symbols with pending news

        # Correlation limits
        self.max_correlated_positions = self.config.get('max_correlated_positions', 2)
        self.correlation_groups = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'AMZN'],
            'crypto_related': ['MARA', 'RIOT', 'COIN', 'MSTR'],
            'ev': ['TSLA', 'RIVN', 'LCID', 'NIO'],
            'meme': ['GME', 'AMC', 'BBBY'],
            'financials': ['JPM', 'BAC', 'GS', 'MS'],
            'energy': ['XOM', 'CVX', 'OXY', 'SLB'],
        }

        # State tracking
        self.daily_stats: Dict[date, DailyStats] = {}
        self.symbol_trades_today: Dict[str, int] = {}
        self.last_loss_time: Optional[datetime] = None
        self.strategy_performance: Dict[str, Dict] = {}  # Strategy -> {wins, losses}
        self.active_positions: Dict[str, str] = {}  # Symbol -> direction
        self.rejected_trades: List[Tuple[datetime, TradeSignal, List[RejectReason]]] = []

    def _get_today_stats(self) -> DailyStats:
        """Get or create today's statistics"""
        today = date.today()
        if today not in self.daily_stats:
            self.daily_stats[today] = DailyStats(date=today)
            # Reset symbol tracking
            self.symbol_trades_today = {}
        return self.daily_stats[today]

    def filter_trade(self, signal: TradeSignal,
                     current_positions: Dict[str, Any] = None,
                     market_data: Optional[Dict] = None) -> FilterResult:
        """
        Main filtering logic. Returns whether trade should be taken.

        Args:
            signal: The trade signal to evaluate
            current_positions: Dict of current open positions
            market_data: Optional OHLCV data for regime detection

        Returns:
            FilterResult with decision and reasoning
        """
        current_positions = current_positions or {}
        reject_reasons = []
        warnings = []
        stats = self._get_today_stats()

        # ============================================
        # HARD BLOCKS - These are non-negotiable
        # ============================================

        # 1. Daily R Target Hit - STOP TRADING
        if stats.r_target_hit:
            reject_reasons.append(RejectReason.DAILY_R_TARGET_HIT)
            logger.info(f"BLOCKED: Daily R target already hit ({stats.total_r:.2f}R)")

        # 2. Daily Loss Limit Hit
        if stats.total_r <= self.daily_r_loss_limit:
            reject_reasons.append(RejectReason.DRAWDOWN_MODE)
            logger.info(f"BLOCKED: Daily loss limit hit ({stats.total_r:.2f}R)")

        # 3. Consecutive Losses - Take a break
        if stats.consecutive_losses >= self.max_consecutive_losses:
            if self.last_loss_time:
                time_since_loss = datetime.now() - self.last_loss_time
                if time_since_loss < timedelta(minutes=self.loss_cooldown_minutes):
                    reject_reasons.append(RejectReason.CONSECUTIVE_LOSSES)
                    logger.info(f"BLOCKED: {stats.consecutive_losses} consecutive losses. Cooling off.")

        # 4. Session Filter
        session_allowed, session_reason = self.session_manager.should_trade("entry")
        if not session_allowed:
            reject_reasons.append(RejectReason.SESSION_BLOCKED)
            logger.info(f"BLOCKED: {session_reason}")

        # 5. Dead Zone
        if self.session_manager.is_dead_zone():
            reject_reasons.append(RejectReason.DEAD_ZONE)
            logger.info("BLOCKED: Dead zone - no trading")

        # 6. Regime Check (if data available)
        if market_data is not None:
            import pandas as pd
            if isinstance(market_data, pd.DataFrame) and len(market_data) > 30:
                regime_state = self.regime_detector.detect_regime(signal.symbol, market_data)

                if regime_state.avoid_trading:
                    reject_reasons.append(RejectReason.CHOPPY_MARKET)
                    logger.info(f"BLOCKED: {signal.symbol} in {regime_state.regime.value} regime")

                # Check strategy vs regime compatibility
                regime_allowed, regime_reason = self.regime_detector.should_trade(
                    signal.symbol,
                    self._classify_strategy(signal.strategy_name)
                )
                if not regime_allowed:
                    reject_reasons.append(RejectReason.REGIME_MISMATCH)
                    warnings.append(regime_reason)

        # 7. News Blackout
        if signal.symbol in self.news_blackout_symbols:
            reject_reasons.append(RejectReason.NEWS_DAY)
            logger.info(f"BLOCKED: {signal.symbol} has pending news")

        # 8. Overtrading Protection
        if stats.trades_taken >= self.max_trades_per_day:
            reject_reasons.append(RejectReason.OVERTRADING)
            logger.info(f"BLOCKED: Max daily trades ({self.max_trades_per_day}) reached")

        # 9. Symbol Trade Limit
        symbol_trades = self.symbol_trades_today.get(signal.symbol, 0)
        if symbol_trades >= self.max_trades_per_symbol:
            reject_reasons.append(RejectReason.OVERTRADING)
            logger.info(f"BLOCKED: Max trades for {signal.symbol} reached")

        # 10. Correlation Check
        correlated_count = self._count_correlated_positions(signal.symbol, current_positions)
        if correlated_count >= self.max_correlated_positions:
            reject_reasons.append(RejectReason.CORRELATED_EXPOSURE)
            logger.info(f"BLOCKED: Too many correlated positions ({correlated_count})")

        # ============================================
        # QUALITY CHECKS - Setup must be good enough
        # ============================================

        # 11. Minimum Confidence
        if signal.confidence < self.min_confidence:
            reject_reasons.append(RejectReason.WEAK_SETUP)
            logger.info(f"BLOCKED: Confidence {signal.confidence:.2f} < {self.min_confidence}")

        # 12. R-Multiple Check
        r_multiple = self._calculate_r_multiple(signal)
        if r_multiple < self.min_r_multiple:
            reject_reasons.append(RejectReason.WEAK_SETUP)
            logger.info(f"BLOCKED: R-multiple {r_multiple:.2f} < {self.min_r_multiple}")

        # 13. Confluence Score
        confluence = self._calculate_confluence(signal)
        if confluence < self.min_confluence_score:
            reject_reasons.append(RejectReason.LOW_CONFLUENCE)
            logger.info(f"BLOCKED: Confluence {confluence} < {self.min_confluence_score}")

        # 14. Strategy Win Rate Filter
        strategy_stats = self.strategy_performance.get(signal.strategy_name, {})
        if strategy_stats.get('total', 0) >= self.min_strategy_trades:
            win_rate = strategy_stats.get('wins', 0) / strategy_stats['total']
            if win_rate < self.min_strategy_win_rate:
                reject_reasons.append(RejectReason.WIN_RATE_FILTER)
                warnings.append(f"Strategy {signal.strategy_name} win rate: {win_rate:.1%}")

        # ============================================
        # CALCULATE QUALITY SCORE
        # ============================================

        quality_score = self._calculate_quality_score(signal, confluence, r_multiple)

        if quality_score < self.min_quality_score:
            if RejectReason.WEAK_SETUP not in reject_reasons:
                reject_reasons.append(RejectReason.WEAK_SETUP)
            logger.info(f"BLOCKED: Quality score {quality_score:.0f} < {self.min_quality_score}")

        # ============================================
        # SIZE ADJUSTMENTS
        # ============================================

        adjusted_size = signal.position_size

        # Reduce size in drawdown mode
        if stats.is_in_drawdown:
            adjusted_size *= self.drawdown_size_reduction
            warnings.append(f"Size reduced {self.drawdown_size_reduction:.0%} (drawdown mode)")

        # Reduce size based on session
        session_mult = self.session_manager.get_position_size_multiplier()
        adjusted_size *= session_mult

        # Reduce size based on regime
        if market_data is not None:
            try:
                import pandas as pd
                if isinstance(market_data, pd.DataFrame) and len(market_data) > 30:
                    regime_state = self.regime_detector.detect_regime(signal.symbol, market_data)
                    adjusted_size *= regime_state.recommended_position_size
            except Exception:
                pass

        # ============================================
        # FINAL DECISION
        # ============================================

        allowed = len(reject_reasons) == 0

        if allowed:
            message = f"APPROVED: {signal.symbol} {signal.direction} via {signal.strategy_name} " \
                      f"(Quality: {quality_score:.0f}, Confluence: {confluence}, R: {r_multiple:.2f})"
            logger.info(message)
        else:
            message = f"REJECTED: {signal.symbol} - " + ", ".join([r.value for r in reject_reasons])
            # Track rejected trades for analysis
            self.rejected_trades.append((datetime.now(), signal, reject_reasons))
            if len(self.rejected_trades) > 100:
                self.rejected_trades = self.rejected_trades[-100:]

        return FilterResult(
            allowed=allowed,
            signal=signal,
            reject_reasons=reject_reasons,
            warnings=warnings,
            adjusted_size=adjusted_size if allowed else None,
            confluence_score=confluence,
            quality_score=quality_score,
            message=message
        )

    def _calculate_r_multiple(self, signal: TradeSignal) -> float:
        """Calculate risk:reward ratio"""
        if signal.direction == 'long':
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
        else:
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profit

        if risk <= 0:
            return 0.0

        return reward / risk

    def _calculate_confluence(self, signal: TradeSignal) -> int:
        """
        Calculate confluence score based on confirming signals.
        More confirming factors = higher quality setup.
        """
        score = 0

        # Base score for the primary signal
        score += 1

        # Check additional confirming signals
        additional = signal.additional_signals

        # Trend alignment
        if additional.get('trend_aligned', False):
            score += 1

        # Support/Resistance confluence
        if additional.get('near_key_level', False):
            score += 1

        # Volume confirmation
        if additional.get('volume_confirmed', False):
            score += 1

        # Multiple timeframe alignment
        if additional.get('htf_aligned', False):  # Higher timeframe
            score += 1

        # Momentum confirmation (RSI, MACD agreement)
        if additional.get('momentum_confirmed', False):
            score += 1

        # Sentiment alignment
        if additional.get('sentiment_aligned', False):
            score += 1

        # ML model agreement
        if additional.get('ml_confirmed', False):
            score += 1

        return score

    def _calculate_quality_score(self, signal: TradeSignal,
                                  confluence: int, r_multiple: float) -> float:
        """
        Calculate overall setup quality score (0-100).
        Combines multiple factors into single score.
        """
        score = 0.0

        # Confidence (0-25 points)
        score += min(signal.confidence * 25, 25)

        # R-Multiple (0-25 points)
        # 1.5R = 15 pts, 2R = 20 pts, 3R+ = 25 pts
        r_score = min(r_multiple * 10, 25)
        score += r_score

        # Confluence (0-25 points)
        # Each confluence point = 5 pts
        score += min(confluence * 5, 25)

        # Session bonus (0-15 points)
        if self.session_manager.is_prime_time():
            score += 15
        elif not self.session_manager.is_dead_zone():
            score += 7

        # Strategy track record (0-10 points)
        strategy_stats = self.strategy_performance.get(signal.strategy_name, {})
        if strategy_stats.get('total', 0) >= self.min_strategy_trades:
            win_rate = strategy_stats.get('wins', 0) / strategy_stats['total']
            score += win_rate * 10

        return min(score, 100)

    def _classify_strategy(self, strategy_name: str) -> str:
        """Classify strategy type for regime compatibility"""
        strategy_lower = strategy_name.lower()

        if any(x in strategy_lower for x in ['trend', 'momentum', 'breakout', 'ma_cross']):
            return 'trend_following'
        elif any(x in strategy_lower for x in ['mean', 'reversion', 'rsi', 'oversold', 'overbought']):
            return 'mean_reversion'
        elif any(x in strategy_lower for x in ['breakout', 'range']):
            return 'breakout'
        else:
            return 'trend_following'  # Default

    def _count_correlated_positions(self, symbol: str,
                                     current_positions: Dict[str, Any]) -> int:
        """Count positions in same correlation group"""
        # Find which group this symbol belongs to
        symbol_group = None
        for group, symbols in self.correlation_groups.items():
            if symbol in symbols:
                symbol_group = group
                break

        if symbol_group is None:
            return 0

        # Count existing positions in same group
        count = 0
        group_symbols = self.correlation_groups[symbol_group]
        for pos_symbol in current_positions.keys():
            if pos_symbol in group_symbols:
                count += 1

        return count

    # ============================================
    # STATE UPDATES
    # ============================================

    def record_trade_result(self, symbol: str, strategy: str,
                            r_result: float, won: bool):
        """
        Record trade result for filtering decisions.

        Args:
            symbol: Trading symbol
            strategy: Strategy name
            r_result: R-multiple result (+2.5R, -1R, etc.)
            won: Whether trade was profitable
        """
        stats = self._get_today_stats()

        stats.trades_taken += 1
        stats.total_r += r_result

        if won:
            stats.wins += 1
            stats.consecutive_losses = 0
        else:
            stats.losses += 1
            stats.consecutive_losses += 1
            self.last_loss_time = datetime.now()

        # Track max drawdown
        if stats.total_r < stats.max_drawdown_r:
            stats.max_drawdown_r = stats.total_r

        # Check drawdown mode
        if stats.total_r <= self.drawdown_threshold:
            stats.is_in_drawdown = True
            logger.warning(f"DRAWDOWN MODE ACTIVATED: {stats.total_r:.2f}R")

        # Check R target hit
        if stats.total_r >= self.daily_r_target:
            stats.r_target_hit = True
            logger.info(f"DAILY R TARGET HIT: {stats.total_r:.2f}R - STOP TRADING")

        # Update symbol tracking
        self.symbol_trades_today[symbol] = self.symbol_trades_today.get(symbol, 0) + 1

        # Update strategy performance
        if strategy not in self.strategy_performance:
            self.strategy_performance[strategy] = {'wins': 0, 'losses': 0, 'total': 0}
        self.strategy_performance[strategy]['total'] += 1
        if won:
            self.strategy_performance[strategy]['wins'] += 1
        else:
            self.strategy_performance[strategy]['losses'] += 1

        logger.info(f"Trade recorded: {symbol} {strategy} = {r_result:+.2f}R "
                   f"(Daily: {stats.total_r:+.2f}R, Streak: {stats.consecutive_losses} losses)")

    def add_news_blackout(self, symbol: str, duration_minutes: int = 60):
        """Block trading on a symbol due to news"""
        self.news_blackout_symbols.append(symbol)
        logger.info(f"NEWS BLACKOUT: {symbol} blocked for {duration_minutes} minutes")

    def remove_news_blackout(self, symbol: str):
        """Remove news blackout for a symbol"""
        if symbol in self.news_blackout_symbols:
            self.news_blackout_symbols.remove(symbol)

    def update_position(self, symbol: str, direction: Optional[str]):
        """Update active position tracking"""
        if direction:
            self.active_positions[symbol] = direction
        elif symbol in self.active_positions:
            del self.active_positions[symbol]

    def get_filter_summary(self) -> Dict:
        """Get summary of filter state for commentary"""
        stats = self._get_today_stats()
        session = self.session_manager.get_session_summary()

        return {
            "daily_r": f"{stats.total_r:+.2f}R",
            "daily_r_target": f"{self.daily_r_target}R",
            "daily_r_limit": f"{self.daily_r_loss_limit}R",
            "r_target_hit": stats.r_target_hit,
            "trades_today": stats.trades_taken,
            "max_trades": self.max_trades_per_day,
            "wins": stats.wins,
            "losses": stats.losses,
            "consecutive_losses": stats.consecutive_losses,
            "in_drawdown_mode": stats.is_in_drawdown,
            "session": session['session_name'],
            "is_prime_time": session['is_prime_time'],
            "is_dead_zone": session['is_dead_zone'],
            "news_blackout_symbols": self.news_blackout_symbols,
            "rejected_today": len([r for r in self.rejected_trades
                                   if r[0].date() == date.today()]),
            "status": self._get_status_message(stats)
        }

    def _get_status_message(self, stats: DailyStats) -> str:
        """Generate status message"""
        if stats.r_target_hit:
            return f"DAILY TARGET HIT ({stats.total_r:+.2f}R) - TRADING PAUSED"
        if stats.total_r <= self.daily_r_loss_limit:
            return f"LOSS LIMIT HIT ({stats.total_r:+.2f}R) - TRADING STOPPED"
        if stats.is_in_drawdown:
            return f"DRAWDOWN MODE - REDUCED SIZE ({stats.total_r:+.2f}R)"
        if stats.consecutive_losses >= self.max_consecutive_losses:
            return f"COOLING OFF - {stats.consecutive_losses} consecutive losses"
        if self.session_manager.is_dead_zone():
            return "DEAD ZONE - NO NEW TRADES"
        if self.session_manager.is_prime_time():
            return "PRIME TIME - ACTIVE"
        return f"ACTIVE - {stats.total_r:+.2f}R today"


# Singleton instance
_no_trade_filter: Optional[NoTradeFilter] = None

def get_no_trade_filter(config: Optional[Dict] = None) -> NoTradeFilter:
    """Get or create singleton NoTradeFilter instance"""
    global _no_trade_filter
    if _no_trade_filter is None:
        _no_trade_filter = NoTradeFilter(config)
    return _no_trade_filter
