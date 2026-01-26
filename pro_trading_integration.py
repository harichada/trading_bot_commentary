"""
Professional Trading Integration Layer
Wraps the existing trading bot with professional-grade filters and management

This module integrates all professional components:
- RegimeDetector: Know what market you're in
- SessionManager: Know when to trade
- NoTradeFilter: Block low-quality setups
- MarketContext: Understand the context
- ExitManager: Professional exit management
- PerformanceTracker: Track R-multiples

Usage:
    from pro_trading_integration import ProTradingWrapper
    wrapper = ProTradingWrapper()

    # Before taking a trade:
    result = wrapper.evaluate_trade_signal(signal, market_data)
    if result.allowed:
        # Take the trade with adjusted parameters
        pass

    # Managing exits:
    exit_signals = wrapper.update_positions(current_prices, market_data)

    # After closing a trade:
    wrapper.record_trade_result(trade_summary)
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
import uuid

# Import all professional modules
from regime_detector import get_regime_detector, RegimeDetector, MarketRegime
from session_manager import get_session_manager, SessionManager, MarketSession
from no_trade_filter import get_no_trade_filter, NoTradeFilter, TradeSignal, FilterResult
from market_context import get_market_context_analyzer, MarketContextAnalyzer
from exit_manager import get_exit_manager, ProExitManager, ExitSignal, ExitReason
from performance_tracker import get_performance_tracker, PerformanceTracker, TradeRecord

logger = logging.getLogger(__name__)


@dataclass
class TradeDecision:
    """Final decision on whether to take a trade"""
    allowed: bool
    symbol: str
    direction: str
    strategy: str

    # Adjusted parameters
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_amount: float

    # Context
    regime: str
    session: str
    quality_score: float
    confluence_score: int

    # Reasoning (for commentary)
    reasons_for: List[str]
    reasons_against: List[str]
    final_recommendation: str

    # R-multiple info
    r_multiple_target: float
    risk_per_share: float


class ProTradingWrapper:
    """
    Professional trading wrapper that integrates all modules.

    This is the main integration point. Use this to:
    1. Evaluate incoming trade signals
    2. Manage open positions
    3. Track performance

    Philosophy: Every trade decision goes through multiple quality filters.
    If ANY filter rejects, the trade is blocked.
    """

    def __init__(self, config_file: str = "pro_trading_config.yaml"):
        self.config = self._load_config(config_file)

        # Initialize all modules with config
        self.regime_detector = get_regime_detector(
            self.config.get('regime_detector', {})
        )
        self.session_manager = get_session_manager(
            self.config.get('session_manager', {})
        )
        self.no_trade_filter = get_no_trade_filter(
            self.config.get('no_trade_filter', {})
        )
        self.market_context = get_market_context_analyzer(
            self.config.get('market_context', {})
        )
        self.exit_manager = get_exit_manager(
            self.config.get('exit_manager', {})
        )
        self.performance_tracker = get_performance_tracker(
            self.config.get('performance_tracker', {})
        )

        # Feature flags
        self.features = self.config.get('features', {})

        # State tracking
        self.active_positions: Dict[str, Dict] = {}
        self.pending_signals: Dict[str, TradeDecision] = {}

        logger.info("ProTradingWrapper initialized with all modules")

    def _load_config(self, config_file: str) -> Dict:
        """Load configuration from YAML file"""
        config_path = Path(config_file)
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded config from {config_file}")
                return config
        else:
            logger.warning(f"Config file {config_file} not found, using defaults")
            return {}

    def evaluate_trade_signal(
        self,
        symbol: str,
        direction: str,
        strategy: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        position_size: float,
        confidence: float,
        market_data: Optional[pd.DataFrame] = None,
        additional_signals: Optional[Dict] = None
    ) -> TradeDecision:
        """
        Evaluate a trade signal through all professional filters.

        This is the main entry point for trade decisions.

        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            strategy: Strategy name that generated the signal
            entry_price: Proposed entry price
            stop_loss: Proposed stop loss
            take_profit: Proposed take profit
            position_size: Proposed position size
            confidence: Signal confidence (0-1)
            market_data: OHLCV DataFrame for context analysis
            additional_signals: Dict of supporting signals for confluence

        Returns:
            TradeDecision with allowed/blocked status and adjusted parameters
        """
        reasons_for = []
        reasons_against = []

        # Default values
        adjusted_size = position_size
        adjusted_stop = stop_loss
        adjusted_target = take_profit

        # Calculate base risk
        if direction == 'long':
            risk_per_share = entry_price - stop_loss
        else:
            risk_per_share = stop_loss - entry_price

        risk_amount = risk_per_share * position_size

        # ================================================================
        # 1. REGIME DETECTION
        # ================================================================
        regime = "unknown"
        regime_allowed = True

        if self.features.get('regime_detector', True) and market_data is not None:
            regime_state = self.regime_detector.detect_regime(symbol, market_data)
            regime = regime_state.regime.value

            # Check if regime allows trading
            strategy_type = self._classify_strategy(strategy)
            regime_allowed, regime_reason = self.regime_detector.should_trade(
                symbol, strategy_type
            )

            if regime_allowed:
                reasons_for.append(f"Regime: {regime} supports {strategy_type}")
            else:
                reasons_against.append(f"Regime: {regime_reason}")

            # Adjust size based on regime
            adjusted_size *= regime_state.recommended_position_size

            # Adjust stops based on regime
            regime_params = self.regime_detector.get_regime_parameters(symbol)
            adjusted_stop = self._adjust_stop_for_regime(
                entry_price, stop_loss, direction, regime_params.stop_loss_multiplier
            )

        # ================================================================
        # 2. SESSION CHECK
        # ================================================================
        session = "unknown"
        session_allowed = True

        if self.features.get('session_manager', True):
            session_state = self.session_manager.get_session_state()
            session = session_state.session.value

            session_allowed, session_reason = self.session_manager.should_trade("entry")

            if session_allowed:
                reasons_for.append(f"Session: {session_state.session_name}")
                if self.session_manager.is_prime_time():
                    reasons_for.append("PRIME TIME - Best trading window")
            else:
                reasons_against.append(f"Session: {session_reason}")

            # Adjust size for session
            adjusted_size *= self.session_manager.get_position_size_multiplier()

        # ================================================================
        # 3. MARKET CONTEXT
        # ================================================================
        context_score = 50
        key_level = None

        if self.features.get('market_context', True) and market_data is not None:
            context = self.market_context.analyze(symbol, market_data, entry_price)

            if direction == 'long':
                context_score = context.long_quality
                key_level = context.nearest_support.price if context.nearest_support else None

                if context.long_quality > 70:
                    reasons_for.append(f"Context: Strong long setup ({context.long_quality:.0f}/100)")
                elif context.long_quality < 40:
                    reasons_against.append(f"Context: Weak long setup ({context.long_quality:.0f}/100)")

                if context.potential_squeeze == "short_squeeze":
                    reasons_for.append("Potential short squeeze detected")
            else:
                context_score = context.short_quality
                key_level = context.nearest_resistance.price if context.nearest_resistance else None

                if context.short_quality > 70:
                    reasons_for.append(f"Context: Strong short setup ({context.short_quality:.0f}/100)")
                elif context.short_quality < 40:
                    reasons_against.append(f"Context: Weak short setup ({context.short_quality:.0f}/100)")

                if context.potential_squeeze == "long_squeeze":
                    reasons_for.append("Potential long squeeze detected")

            # Use context for better stop placement
            if key_level and self.features.get('use_context_for_stops', True):
                context_stop = self._calculate_context_stop(
                    entry_price, key_level, direction
                )
                # Use the wider of the two stops for safety
                if direction == 'long':
                    adjusted_stop = min(adjusted_stop, context_stop)
                else:
                    adjusted_stop = max(adjusted_stop, context_stop)

        # ================================================================
        # 4. NO TRADE FILTER (The Gatekeeper)
        # ================================================================
        filter_result = None

        if self.features.get('no_trade_filter', True):
            # Build trade signal for filter
            trade_signal = TradeSignal(
                symbol=symbol,
                direction=direction,
                strategy_name=strategy,
                confidence=confidence,
                entry_price=entry_price,
                stop_loss=adjusted_stop,
                take_profit=take_profit,
                risk_amount=risk_amount,
                position_size=adjusted_size,
                additional_signals=additional_signals or {}
            )

            filter_result = self.no_trade_filter.filter_trade(
                trade_signal,
                current_positions=self.active_positions,
                market_data=market_data
            )

            if filter_result.allowed:
                reasons_for.append(f"Quality gate: PASSED (Score: {filter_result.quality_score:.0f})")
            else:
                for reason in filter_result.reject_reasons:
                    reasons_against.append(f"Filter: {reason.value}")

            # Use filter's adjusted size
            if filter_result.adjusted_size:
                adjusted_size = filter_result.adjusted_size

        # ================================================================
        # 5. PERFORMANCE-BASED FILTERING
        # ================================================================
        if self.features.get('performance_tracker', True):
            # Check strategy performance
            strategy_ok, strategy_reason = self.performance_tracker.should_use_strategy(strategy)
            if not strategy_ok:
                reasons_against.append(f"Strategy: {strategy_reason}")

            # Check symbol performance
            symbol_ok, symbol_reason = self.performance_tracker.should_trade_symbol(symbol)
            if not symbol_ok:
                reasons_against.append(f"Symbol: {symbol_reason}")

        # ================================================================
        # FINAL DECISION
        # ================================================================
        # Trade is allowed only if ALL filters pass
        allowed = (
            regime_allowed and
            session_allowed and
            (filter_result is None or filter_result.allowed) and
            len(reasons_against) == 0
        )

        # Calculate R-multiple target
        if direction == 'long':
            adjusted_risk = entry_price - adjusted_stop
            r_target = (take_profit - entry_price) / adjusted_risk if adjusted_risk > 0 else 0
        else:
            adjusted_risk = adjusted_stop - entry_price
            r_target = (entry_price - take_profit) / adjusted_risk if adjusted_risk > 0 else 0

        # Generate recommendation
        if allowed:
            recommendation = self._generate_approved_recommendation(
                symbol, direction, strategy, regime, session,
                filter_result.quality_score if filter_result else 50,
                r_target
            )
        else:
            recommendation = self._generate_rejected_recommendation(
                symbol, direction, reasons_against
            )

        # Log decision
        self._log_decision(
            symbol, direction, strategy, allowed,
            reasons_for, reasons_against, regime, session
        )

        return TradeDecision(
            allowed=allowed,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_price=entry_price,
            stop_loss=adjusted_stop,
            take_profit=adjusted_target,
            position_size=adjusted_size,
            risk_amount=adjusted_risk * adjusted_size if allowed else 0,
            regime=regime,
            session=session,
            quality_score=filter_result.quality_score if filter_result else 50,
            confluence_score=filter_result.confluence_score if filter_result else 0,
            reasons_for=reasons_for,
            reasons_against=reasons_against,
            final_recommendation=recommendation,
            r_multiple_target=r_target,
            risk_per_share=adjusted_risk if allowed else 0
        )

    def open_position(
        self,
        decision: TradeDecision,
        actual_fill_price: Optional[float] = None,
        atr: Optional[float] = None
    ) -> Dict:
        """
        Open a position after trade is approved.

        Call this after getting an allowed TradeDecision and receiving a fill.

        Args:
            decision: The approved TradeDecision
            actual_fill_price: Actual fill price (may differ from entry_price)
            atr: Current ATR for the symbol

        Returns:
            Position state dict
        """
        if not decision.allowed:
            raise ValueError("Cannot open position for rejected trade")

        fill_price = actual_fill_price or decision.entry_price

        # Open in exit manager
        position_state = self.exit_manager.open_position(
            symbol=decision.symbol,
            direction=decision.direction,
            entry_price=fill_price,
            position_size=decision.position_size,
            stop_loss=decision.stop_loss,
            atr=atr
        )

        # Track in our positions
        self.active_positions[decision.symbol] = {
            'direction': decision.direction,
            'entry_price': fill_price,
            'position_size': decision.position_size,
            'strategy': decision.strategy,
            'regime_at_entry': decision.regime,
            'session_at_entry': decision.session,
            'quality_score': decision.quality_score,
            'entry_time': datetime.now(),
            'trade_id': str(uuid.uuid4())
        }

        # Record trade start in session manager
        if self.features.get('session_manager', True):
            self.session_manager.record_trade()

        logger.info(f"POSITION OPENED: {decision.symbol} {decision.direction} "
                   f"@ ${fill_price:.2f}, Size: {decision.position_size}, "
                   f"Stop: ${decision.stop_loss:.2f}")

        return self.active_positions[decision.symbol]

    def update_positions(
        self,
        current_prices: Dict[str, float],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        atrs: Optional[Dict[str, float]] = None
    ) -> List[ExitSignal]:
        """
        Update all open positions and check for exits.

        Call this on each price update.

        Args:
            current_prices: Dict of symbol -> current price
            market_data: Optional dict of symbol -> OHLCV DataFrame
            atrs: Optional dict of symbol -> current ATR

        Returns:
            List of ExitSignals for positions that need to be closed
        """
        all_exit_signals = []

        for symbol in list(self.active_positions.keys()):
            if symbol not in current_prices:
                continue

            price = current_prices[symbol]
            atr = atrs.get(symbol) if atrs else None
            data = market_data.get(symbol) if market_data else None

            # Check for regime change
            if self.features.get('regime_detector', True) and data is not None:
                regime_state = self.regime_detector.detect_regime(symbol, data)
                if regime_state.avoid_trading and regime_state.regime == MarketRegime.CHOPPY:
                    # Regime turned choppy - consider tightening stops
                    logger.warning(f"{symbol}: Regime turned CHOPPY - tightening management")

            # Update in exit manager
            exit_signals = self.exit_manager.update_position(
                symbol, price, atr, data
            )

            all_exit_signals.extend(exit_signals)

        return all_exit_signals

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: ExitReason,
        partial_size: Optional[float] = None
    ) -> Optional[Dict]:
        """
        Close a position (fully or partially).

        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_reason: Reason for exit
            partial_size: If provided, only close this many shares

        Returns:
            Trade summary dict
        """
        if symbol not in self.active_positions:
            logger.warning(f"Attempted to close unknown position: {symbol}")
            return None

        position = self.active_positions[symbol]

        # Close in exit manager
        summary = self.exit_manager.close_position(symbol, exit_reason, exit_price)

        if summary is None:
            return None

        # Record in performance tracker
        if self.features.get('performance_tracker', True):
            trade_record = TradeRecord(
                trade_id=position['trade_id'],
                symbol=symbol,
                direction=position['direction'],
                strategy=position['strategy'],
                setup_type=position['strategy'],  # Could be more specific
                entry_price=position['entry_price'],
                exit_price=exit_price,
                stop_loss=summary['initial_stop'],
                risk_per_share=summary['risk_per_share'],
                r_result=summary['total_r'],
                max_favorable_r=summary['highest_r'],
                max_adverse_r=0,  # Would need more tracking
                regime=position['regime_at_entry'],
                session=position['session_at_entry'],
                quality_score=position['quality_score'],
                confluence_score=0,  # Would need more tracking
                entry_time=position['entry_time'],
                exit_time=datetime.now(),
                bars_held=summary['bars_in_trade'],
                exit_reason=exit_reason.value,
                position_size=position['position_size'],
                dollar_risk=summary['risk_per_share'] * position['position_size'],
                dollar_pnl=(exit_price - position['entry_price']) * position['position_size']
                          if position['direction'] == 'long'
                          else (position['entry_price'] - exit_price) * position['position_size']
            )
            self.performance_tracker.record_trade(trade_record)

        # Update no_trade_filter
        if self.features.get('no_trade_filter', True):
            won = summary['total_r'] > 0
            self.no_trade_filter.record_trade_result(
                symbol, position['strategy'], summary['total_r'], won
            )

        # Remove from active positions
        del self.active_positions[symbol]

        logger.info(f"POSITION CLOSED: {symbol} @ ${exit_price:.2f}, "
                   f"R: {summary['total_r']:+.2f}, Reason: {exit_reason.value}")

        return summary

    def get_status_summary(self) -> Dict:
        """Get comprehensive status summary for commentary"""
        # Session status
        session = self.session_manager.get_session_summary() if self.features.get('session_manager') else {}

        # Filter status
        filter_status = self.no_trade_filter.get_filter_summary() if self.features.get('no_trade_filter') else {}

        # Overall stats
        overall = self.performance_tracker.get_overall_stats() if self.features.get('performance_tracker') else {}

        # Position summaries
        positions = self.exit_manager.get_all_positions_summary()

        return {
            'timestamp': datetime.now().isoformat(),
            'session': session,
            'filter_status': filter_status,
            'performance': overall,
            'active_positions': positions,
            'active_position_count': len(self.active_positions),
            'is_trading_allowed': self._is_trading_allowed()
        }

    def _is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is currently allowed"""
        reasons = []

        # Check session
        if self.features.get('session_manager', True):
            allowed, reason = self.session_manager.should_trade("entry")
            if not allowed:
                return False, reason

        # Check filter status
        if self.features.get('no_trade_filter', True):
            status = self.no_trade_filter.get_filter_summary()
            if status.get('r_target_hit'):
                return False, "Daily R target hit - trading paused"
            if status.get('in_drawdown_mode'):
                reasons.append("Drawdown mode - reduced size")

        return True, "Trading allowed" + (f" ({', '.join(reasons)})" if reasons else "")

    def _classify_strategy(self, strategy: str) -> str:
        """Classify strategy type for regime compatibility"""
        strategy_lower = strategy.lower()
        if any(x in strategy_lower for x in ['trend', 'momentum', 'breakout', 'ma_cross']):
            return 'trend_following'
        elif any(x in strategy_lower for x in ['mean', 'reversion', 'rsi', 'oversold', 'overbought']):
            return 'mean_reversion'
        elif any(x in strategy_lower for x in ['breakout', 'range']):
            return 'breakout'
        return 'trend_following'

    def _adjust_stop_for_regime(self, entry: float, stop: float,
                                 direction: str, multiplier: float) -> float:
        """Adjust stop loss based on regime multiplier"""
        distance = abs(entry - stop)
        adjusted_distance = distance * multiplier

        if direction == 'long':
            return entry - adjusted_distance
        else:
            return entry + adjusted_distance

    def _calculate_context_stop(self, entry: float, key_level: float,
                                 direction: str) -> float:
        """Calculate stop based on key level"""
        buffer = 0.002  # 0.2% buffer

        if direction == 'long':
            # Stop below support
            return key_level * (1 - buffer)
        else:
            # Stop above resistance
            return key_level * (1 + buffer)

    def _generate_approved_recommendation(
        self, symbol: str, direction: str, strategy: str,
        regime: str, session: str, quality: float, r_target: float
    ) -> str:
        """Generate commentary for approved trade"""
        return (
            f"APPROVED: {symbol} {direction.upper()} via {strategy}. "
            f"Regime: {regime}, Session: {session}, Quality: {quality:.0f}/100, "
            f"R-Target: {r_target:.1f}. This setup meets all professional criteria."
        )

    def _generate_rejected_recommendation(
        self, symbol: str, direction: str, reasons: List[str]
    ) -> str:
        """Generate commentary for rejected trade"""
        reason_str = "; ".join(reasons[:3])  # Top 3 reasons
        return (
            f"BLOCKED: {symbol} {direction.upper()} signal rejected. "
            f"Reasons: {reason_str}. "
            f"A pro sits out when conditions aren't right."
        )

    def _log_decision(
        self, symbol: str, direction: str, strategy: str, allowed: bool,
        reasons_for: List[str], reasons_against: List[str],
        regime: str, session: str
    ):
        """Log trade decision for analysis"""
        status = "APPROVED" if allowed else "REJECTED"
        logger.info(
            f"Trade Decision: {status} | {symbol} {direction} ({strategy}) | "
            f"Regime: {regime} | Session: {session} | "
            f"For: {len(reasons_for)} | Against: {len(reasons_against)}"
        )

        if self.config.get('logging', {}).get('log_filter_reasons', True):
            for reason in reasons_for:
                logger.debug(f"  + {reason}")
            for reason in reasons_against:
                logger.debug(f"  - {reason}")


# Singleton instance
_wrapper: Optional[ProTradingWrapper] = None

def get_pro_trading_wrapper(config_file: str = "pro_trading_config.yaml") -> ProTradingWrapper:
    """Get or create singleton ProTradingWrapper instance"""
    global _wrapper
    if _wrapper is None:
        _wrapper = ProTradingWrapper(config_file)
    return _wrapper
