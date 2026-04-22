import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np

from core.models import CommentaryType, TradingSignal
from core.commentary import TradingCommentary

logger = logging.getLogger('TradingBot')


class Config:
    """Lazy import of Config from main module"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            from trading_bot_commentary_updated import Config as _Config
            cls._instance = _Config()
        return cls._instance


class RiskManagerWithCommentary:
    """Risk manager that explains its decisions"""

    def __init__(self, account_balance: float, commentary_system):
        self.account_balance = account_balance
        self.schwab_daily_pnl = 0  # ONLY use Schwab P&L
        self.consecutive_losses = 0
        self.positions = {}
        self.max_portfolio_heat = 0.06
        self.margin_call = False
        self.buying_power = account_balance * 0.5
        self.commentary = commentary_system
        self.last_schwab_sync = None

    def sync_with_schwab_data(self, schwab_account_info: Dict[str, float]):
        """Sync risk manager with actual Schwab account data"""
        if schwab_account_info:
            # Update with real Schwab data ONLY
            self.schwab_daily_pnl = schwab_account_info.get('day_pnl', 0)
            self.buying_power = schwab_account_info.get('buying_power', self.buying_power)
            self.account_balance = schwab_account_info.get('balance', self.account_balance)

            self.last_schwab_sync = datetime.now()

            # Log the sync
            logger.info(f"Risk Manager synced with Schwab: P&L=${self.schwab_daily_pnl:.2f}, "
                       f"Buying Power=${self.buying_power:.2f}, Balance=${self.account_balance:.2f}")

    async def calculate_position_size_with_commentary(self, signal, current_price: float) -> int:
        """Calculate position size with detailed explanation"""
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.RISK_ASSESSMENT,
            symbol=signal.symbol,
            title=f"💰 Calculating Position Size",
            message="Let me determine the appropriate position size based on risk management rules...",
            importance=7
        ))

        # Risk per share
        risk_per_share = abs(current_price - signal.stop_loss)
        if risk_per_share == 0:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"⚠️ Invalid Stop Loss",
                message="Stop loss is at the same price as entry. Cannot calculate position size.",
                importance=8
            ))
            return 0

        # Maximum risk amount — uses ATR-scaled RISK_PER_TRADE_PCT (default 1%).
        # Strategies now set stop_loss via ATR, so risk_per_share = ATR_mult × ATR.
        # This means: shares = (equity × 1%) / (1.5 × ATR), naturally giving
        # fewer shares for volatile stocks and more for calm ones.
        max_risk_pct = Config().RISK_PER_TRADE_PCT or 0.01
        max_risk_amount = self.account_balance * max_risk_pct

        # Sanity floor: stop_distance must be at least 0.5% of price
        # to prevent absurd position sizes from ATR=0 or stale data.
        min_stop_distance = current_price * 0.005
        risk_per_share = max(risk_per_share, min_stop_distance)
        # Adjust based on recent performance
        if hasattr(self, 'trade_history') and len(self.trade_history) >= 5:
            recent_trades = self.trade_history[-5:]
            wins = sum(1 for t in recent_trades if t.get('pnl', 0) > 0)
            win_rate = wins / 5

            if win_rate >= 0.8:
                max_risk_amount *= 1.3
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"🔥 Hot Streak Adjustment",
                    message=f"Recent win rate {win_rate:.0%}, increasing position size",
                    importance=6
                ))
            elif win_rate <= 0.2:
                max_risk_amount *= 0.5
                self.commentary.add_commentary(TradingCommentary(
                    timestamp=datetime.now(),
                    type=CommentaryType.RISK_ASSESSMENT,
                    symbol=signal.symbol,
                    title=f"❄️ Cold Streak Protection",
                    message=f"Recent win rate {win_rate:.0%}, reducing position size",
                    importance=6
                ))
        self.commentary.add_commentary(TradingCommentary(
            timestamp=datetime.now(),
            type=CommentaryType.RISK_ASSESSMENT,
            symbol=signal.symbol,
            title=f"📊 Risk Calculation",
            message=f"With {max_risk_pct:.1%} risk per trade on ${self.account_balance:,.0f} account",
            data={
                'max_risk_amount': max_risk_amount,
                'risk_per_share': risk_per_share,
                'stop_distance_percent': (risk_per_share / current_price) * 100
            },
            importance=6
        ))

        # Calculate base position size
        position_size = int(max_risk_amount / risk_per_share)

        # Kelly-inspired bet sizing: scale position by signal confidence.
        # For our 2:1 R:R (ATR_REWARD_RISK_RATIO=2.0):
        #   kelly_fraction = 1.5 × win_prob - 0.5
        # Maps: 0.50 → 0.25 (quarter size), 0.65 → 0.475, 0.75 → 0.625
        # Floor at 0.25 to avoid zero-size on marginal signals.
        confidence = getattr(signal, 'confidence', 0.5) or 0.5
        rr = Config().ATR_REWARD_RISK_RATIO or 2.0
        kelly = confidence - (1.0 - confidence) / rr
        kelly = max(kelly, 0.25)  # floor: always take at least 25%
        kelly = min(kelly, 1.0)   # cap: never exceed base size
        position_size = int(position_size * kelly)
        if position_size < 1:
            position_size = 1

        # Audit log — ATR, stop distance, shares, dollar risk, kelly
        atr_val = signal.reasoning.get('atr') if hasattr(signal, 'reasoning') and signal.reasoning else None
        atr_mult_val = signal.reasoning.get('atr_mult') if hasattr(signal, 'reasoning') and signal.reasoning else None
        dollar_risk = position_size * risk_per_share
        logger.info(
            "position_sizing symbol=%s price=%.2f atr=%s atr_mult=%s "
            "stop_dist=%.2f shares=%d notional=%.0f dollar_risk=%.2f "
            "equity=%.0f risk_pct=%.4f confidence=%.2f kelly=%.3f",
            signal.symbol, current_price,
            round(atr_val, 3) if atr_val else "n/a",
            atr_mult_val or "n/a",
            risk_per_share, position_size,
            position_size * current_price, dollar_risk,
            self.account_balance, max_risk_pct,
            confidence, kelly,
        )

        # v-margin-sizing-2026-04-22: cap combines two limits:
        #   (a) MAX_POSITION_VALUE     — absolute $ ceiling (legacy, $10k default)
        #   (b) buying_power × PCT     — scales with margin (15% of BP default)
        # The tighter of the two wins. On a $27k/$71k BP account with default
        # settings, (b) = $10,650 so (b) is slightly larger than (a) — bump
        # trading.max_position_value in Config.yaml to unlock (b) fully.
        abs_cap = Config().MAX_POSITION_VALUE or 10000
        bp_cap = self.buying_power * Config().MAX_POSITION_VALUE_BP_PCT
        max_position_value = min(abs_cap, bp_cap) if bp_cap > 0 else abs_cap

        min_position_size = Config().MIN_POSITION_SIZE or 1  # Default 1 share
        position_value = position_size * current_price
        if position_value > max_position_value:
            old_size = position_size
            position_size = int(max_position_value / current_price)

            cap_source = "buying_power" if bp_cap < abs_cap else "abs_ceiling"
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⚠️ Position Size Capped",
                message=f"Reduced from {old_size} to {position_size} shares "
                        f"(max ${max_position_value:,.0f} via {cap_source})",
                importance=7
            ))

        # Check buying power
        if position_value > self.buying_power:
            old_size = position_size
            position_size = int(self.buying_power * 0.95 / current_price)

            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"⚠️ Position Size Adjusted",
                message=f"Reduced position from {old_size} to {position_size} shares due to buying power constraints",
                data={
                    'required_capital': old_size * current_price,
                    'available_buying_power': self.buying_power,
                    'adjusted_capital': position_size * current_price
                },
                importance=7
            ))

        # Ensure minimum position size
        if position_size < min_position_size:
            position_size = 0
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.WARNING,
                symbol=signal.symbol,
                title=f"❌ Position Too Small",
                message=f"Calculated size below minimum ({min_position_size} shares)",
                importance=8
            ))

        # Final position size
        if position_size > 0:
            self.commentary.add_commentary(TradingCommentary(
                timestamp=datetime.now(),
                type=CommentaryType.RISK_ASSESSMENT,
                symbol=signal.symbol,
                title=f"✅ Position Size Determined",
                message=f"Will trade {position_size} shares, risking ${position_size * risk_per_share:.2f}",
                data={
                    'shares': position_size,
                    'total_value': position_size * current_price,
                    'max_loss': position_size * risk_per_share,
                    'max_gain': position_size * (signal.take_profit - current_price),
                    'risk_reward_ratio': (signal.take_profit - current_price) / risk_per_share
                },
                confidence=0.9,
                importance=8
            ))

        return position_size

    def check_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed"""
        if self.margin_call:
            return False, "Margin call active - resolve before trading"

        if self.buying_power < 100:
            return False, f"Insufficient buying power: ${self.buying_power:.2f}"

        # ONLY use Schwab's P&L
        if self.schwab_daily_pnl <= -Config().MAX_DAILY_LOSS * self.account_balance:
            return False, f"Daily loss limit exceeded (P&L: ${self.schwab_daily_pnl:.2f})"

        if self.consecutive_losses >= Config().MAX_CONSECUTIVE_LOSSES:
            return False, "Max consecutive losses reached"

        return True, "Trading allowed"
