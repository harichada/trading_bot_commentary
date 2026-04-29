import os
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigManager:
    """Manages configuration loading and validation"""

    def __init__(self, config_path: str = "Config().yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file or create default"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            config = self._get_default_config()
            self._save_config(config)
            return config

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'schwab': {
                'callback_url': "https://127.0.0.1",
                'token_path': "token_1.json"
            },
            'trading': {
                'ml_prediction_enabled': True,
                'max_risk_per_trade': 0.02,
                'min_risk_reward_ratio': 2.0,
                'max_daily_loss': 0.05,
                'max_consecutive_losses': 3,
                'extended_hours': {
                    'allow_premarket': False,  # DANGER: Wide spreads, low liquidity
                    'allow_afterhours': False,  # DANGER: Wide spreads, low liquidity
                    'use_limit_orders': True,  # Always use limit orders in extended hours
                    'premarket_start': "04:00",  # 4:00 AM ET
                    'market_open': "09:30",      # 9:30 AM ET
                    'market_close': "16:00",      # 4:00 PM ET
                    'afterhours_end': "20:00"    # 8:00 PM ET
                },
                'position_size_kelly_fraction': 0.25,
                'max_positions': 5,
                'reserve_cash_percent': 0.1,
                'min_position_size': 1,
                'max_position_value': 10000,
                'min_buying_power': 100,
                'limit_order_slippage': 0.001,
                'default_stop_loss_pct': 0.05,
                'default_take_profit_pct': 0.10,
                'max_position_value_pct': 0.25
            },
            'commentary': {
                'enabled': True,
                'detail_level': "verbose",
                'max_history': 100
            },
            'technical_analysis': {
                'timeframes': ['1min', '5min', '15min', '1hour', '1day'],
                'fibonacci_levels': [0.236, 0.382, 0.5, 0.618, 0.786]
            },
            'order_management': {
                'auto_cancel_existing_orders': True,
                'require_close_confirmation': True,
                'confirm_only_losses': False,
                'confirm_threshold_percent': 5
            },
            'paths': {
                'trade_journal': "trade_journal.json",
                'model': "trading_model.pkl",
                'log': "trading_bot.log",
                'commentary_log': "trading_commentary.json",
                'backtest_results': "backtest_results.json"
            },
            'backtesting': {
                'enabled': True,
                'start_date': "2023-01-01",
                'end_date': "2024-01-01",
                'initial_capital': 100000,
                'commission': 0.005,
                'slippage': 0.001
            },
            'performance_metrics': {
                'calculate_sharpe': True,
                'calculate_sortino': True,
                'calculate_max_drawdown': True,
                'risk_free_rate': 0.02
            },
            'error_recovery': {
                'max_retries': 3,
                'retry_delay': 1.0,
                'circuit_breaker_enabled': True,
                'max_daily_loss_threshold': 0.10,
                'emergency_stop_loss': 0.15
            }
        }

    def _save_config(self, config: dict):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def _validate_config(self):
        """Validate configuration values"""
        required_sections = ['schwab', 'trading', 'commentary', 'technical_analysis']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate trading parameters are within safe ranges
        trading = self.config.get('trading', {})
        validations = [
            ('max_risk_per_trade', 0.001, 0.10, 'Max risk per trade must be 0.1%-10%'),
            ('min_risk_reward_ratio', 0.5, 10.0, 'Risk/reward ratio must be 0.5-10.0'),
            ('max_daily_loss', 0.01, 0.20, 'Max daily loss must be 1%-20%'),
            ('max_positions', 1, 50, 'Max positions must be 1-50'),
            ('reserve_cash_percent', 0.0, 0.50, 'Reserve cash must be 0%-50%'),
        ]
        for key, min_val, max_val, msg in validations:
            val = trading.get(key)
            if val is not None and not (min_val <= val <= max_val):
                raise ValueError(f"Config validation error: {msg} (got {val})")

    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def update(self, key: str, value):
        """Update configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self._save_config(self.config)

# Initialize configuration
config_manager = ConfigManager()

# ============================================================================
# TRADING LOSS CIRCUIT BREAKER
# ============================================================================
# Note: This is distinct from circuit_breaker.py which handles API failures.
# This class monitors daily P&L and halts trading on excessive losses.

class TradingLossBreaker:
    """Circuit breaker that halts trading when daily losses exceed thresholds"""

    def __init__(self, max_daily_loss: float = 0.10, emergency_stop: float = 0.15):
        self.max_daily_loss = max_daily_loss
        self.emergency_stop = emergency_stop
        self.daily_pnl = 0.0
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None

    def update_daily_pnl(self, pnl: float, account_balance: float = 100000):
        """Update daily P&L and check circuit breaker"""
        self.daily_pnl = pnl

        if self.daily_pnl >= 0:
            return

        daily_loss_ratio = abs(self.daily_pnl) / account_balance

        if daily_loss_ratio >= self.emergency_stop:
            self._trip("EMERGENCY_STOP", daily_loss_ratio)
        elif daily_loss_ratio >= self.max_daily_loss:
            self._trip("MAX_DAILY_LOSS", daily_loss_ratio)

    def _trip(self, reason: str, loss_ratio: float):
        """Trip the circuit breaker"""
        self.is_tripped = True
        self.trip_time = datetime.now()
        self.reset_time = self.trip_time + timedelta(hours=24)
        logger.critical(f"Trading loss breaker tripped: {reason} - Loss ratio: {loss_ratio:.2%}")

    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        if not self.is_tripped:
            return True
        if datetime.now() >= self.reset_time:
            self.reset()
            return True
        return False

    def reset(self):
        """Reset the circuit breaker"""
        self.is_tripped = False
        self.trip_time = None
        self.reset_time = None
        logger.info("Trading loss breaker reset")

# Backward compatibility alias
CircuitBreaker = TradingLossBreaker

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

class Config:
    """Central configuration for the trading bot"""
    # Use config manager instead of hardcoded values
    def __init__(self):
        self.manager = config_manager

    @property
    def SCHWAB_API_KEY(self):
        return os.getenv("SCHWAB_API_KEY")

    @property
    def SCHWAB_APP_SECRET(self):
        # Support both SCHWAB_SECRET and SCHWAB_APP_SECRET for compatibility
        return os.getenv("SCHWAB_SECRET") or os.getenv("SCHWAB_APP_SECRET")

    @property
    def SCHWAB_CALLBACK_URL(self):
        return self.manager.get('schwab.callback_url', 'https://127.0.0.1')

    @property
    def SCHWAB_TOKEN_PATH(self):
        return Path(self.manager.get('schwab.token_path', 'token_1.json'))

    @property
    def SCHWAB_ACCOUNT_NUMBER(self):
        return os.getenv("SCHWAB_ACCOUNT_NUMBER")

    @property
    def MAX_RISK_PER_TRADE(self):
        return self.manager.get('trading.max_risk_per_trade', 0.02)

    @property
    def MIN_RISK_REWARD_RATIO(self):
        return self.manager.get('trading.min_risk_reward_ratio', 2.0)

    @property
    def MAX_DAILY_LOSS(self):
        return self.manager.get('trading.max_daily_loss', 0.05)

    @property
    def MAX_CONSECUTIVE_LOSSES(self):
        return self.manager.get('trading.max_consecutive_losses', 3)

    @property
    def POSITION_SIZE_KELLY_FRACTION(self):
        return self.manager.get('trading.position_size_kelly_fraction', 0.25)

    @property
    def COMMENTARY_ENABLED(self):
        return self.manager.get('commentary.enabled', True)

    @property
    def COMMENTARY_DETAIL_LEVEL(self):
        return self.manager.get('commentary.detail_level', 'verbose')

    @property
    def MAX_COMMENTARY_HISTORY(self):
        return self.manager.get('commentary.max_history', 100)

    @property
    def TIMEFRAMES(self):
        return self.manager.get('technical_analysis.timeframes')

    @property
    def FIBONACCI_LEVELS(self):
        return self.manager.get('technical_analysis.fibonacci_levels')

    @property
    def TRADE_JOURNAL_PATH(self):
        return Path(self.manager.get('paths.trade_journal'))

    @property
    def MODEL_PATH(self):
        return Path(self.manager.get('paths.model'))

    @property
    def LOG_PATH(self):
        return Path(self.manager.get('paths.log'))

    @property
    def COMMENTARY_LOG_PATH(self):
        return Path(self.manager.get('paths.commentary_log'))

    @property
    def AUTO_CANCEL_EXISTING_ORDERS(self):
        return self.manager.get('order_management.auto_cancel_existing_orders')

    @property
    def MAX_POSITIONS(self):
        return self.manager.get('trading.max_positions', 5)

    @property
    def RESERVE_CASH_PERCENT(self):
        return self.manager.get('trading.reserve_cash_percent', 0.10)

    @property
    def MIN_POSITION_SIZE(self):
        return self.manager.get('trading.min_position_size', 1)

    @property
    def MAX_POSITION_VALUE(self):
        # v-margin-sizing-2026-04-22: raised default from $10k → $25k so the
        # new buying-power-based cap (MAX_POSITION_VALUE_BP_PCT, 15% of BP)
        # becomes the binding constraint on typical margin accounts. This
        # field now acts as an absolute safety ceiling; the BP cap does the
        # real adaptive work. Legacy config files pinning this to 10000
        # still work — they'll just bind tighter than the BP cap.
        return self.manager.get('trading.max_position_value', 25000)

    @property
    def MIN_BUYING_POWER(self):
        return self.manager.get('trading.min_buying_power', 100)

    @property
    def REQUIRE_CLOSE_CONFIRMATION(self):
        return self.manager.get('order_management.require_close_confirmation', True)

    @property
    def CONFIRM_ONLY_LOSSES(self):
        return self.manager.get('order_management.confirm_only_losses', False)

    @property
    def CONFIRM_THRESHOLD_PERCENT(self):
        return self.manager.get('order_management.confirm_threshold_percent', 5)

    @property
    def ML_PREDICTION_ENABLED(self):
        return self.manager.get('trading.ml_prediction_enabled', True)

    @property
    def ML_VETO_CONFIDENCE(self) -> float:
        """ML confidence (0-1) at or above which a disagreeing ML signal vetoes
        the trade. Default 0.65 — set higher for fewer vetoes, lower for stricter."""
        return float(self.manager.get('trading.ml_veto_confidence', 0.65))

    @property
    def WARMUP_MINUTES_BEFORE_OPEN(self) -> int:
        """Minutes before regular market open (09:30 ET) to wake the bot so
        it can scan the watchlist and build indicator context before the
        first tradable bar. New entries are still blocked until regular
        hours by the signal-router gate.

        Default 30 — set to 0 to disable warmup."""
        return int(self.manager.get('trading.warmup_minutes_before_open', 30))

    @property
    def ENABLE_NEWS_VERIFIER(self) -> bool:
        """v-news-verifier-toggle-2026-04-29: enable the fresh-news
        re-verification gate that was added 2026-04-28.

        Default False — disabled per user request. The gate's veto
        decisions felt opaque ("vetoed for fresh_sentiment_against_signal_avg
        =-0.137" doesn't tell you whether that was right or wrong without
        the shadow-tracker scorecard). When disabled, news_strategy fires
        on cached RSS sentiment alone — same behaviour as before the
        verifier was introduced.

        The shadow-tracker (bot_shadow_news_vetoes table) keeps
        accumulating data when the verifier IS enabled, so toggling this
        back on later still has historical evidence to compare against.

        Set True in Config.yaml to re-enable."""
        return bool(self.manager.get('trading.enable_news_verifier', False))

    @property
    def WATCHLIST_SIZE(self) -> int:
        """v-watchlist-size-2026-04-29: max symbols in the dynamic watchlist.

        Default 20 (was effectively 5-8 due to layered caps). Each cycle
        the bot screens hundreds of symbols, scores them, and picks this
        many to actually analyse and trade. Larger = more opportunities;
        smaller = more focused. Tradeoff: each symbol adds ~2-3 seconds
        per analysis cycle and N×N computation for correlation guard.

        Sweet spot for a 5-min bar bot with max_positions=5 is ~20-25 —
        gives the bot room to find good setups without thrashing CPU."""
        return int(self.manager.get('trading.watchlist_size', 20))

    @property
    def ENABLE_THESIS_REVALIDATION(self) -> bool:
        """v-thesis-revalidate-2026-04-28: re-verify both news + indicators
        on positions older than THESIS_REVALIDATION_AGE_MIN. Closes the
        position when BOTH the news thesis AND the indicator thesis have
        broken (single-signal failure isn't enough — that's proactive_exit).

        Default False = SHADOW MODE: log what would have fired without
        actually closing positions. Flip True after a day of shadow data
        confirms the gate fires appropriately."""
        return bool(self.manager.get('trading.enable_thesis_revalidation', False))

    @property
    def THESIS_REVALIDATION_SHADOW_MODE(self) -> bool:
        """When True (default), thesis re-validation logs the would-be
        decision but does NOT close the position. Set False to enable
        actual exits. Has no effect when ENABLE_THESIS_REVALIDATION=False."""
        return bool(self.manager.get('trading.thesis_revalidation_shadow_mode', True))

    @property
    def THESIS_REVALIDATION_AGE_MIN(self) -> int:
        """Position age in minutes before the first thesis re-check fires.
        Default 30 — gives the trade time to develop without harvesting
        winners that just need patience."""
        return int(self.manager.get('trading.thesis_revalidation_age_min', 30))

    @property
    def THESIS_REVALIDATION_INTERVAL_MIN(self) -> int:
        """Minimum minutes between thesis re-checks for the same position.
        Default 15 — prevents per-tick spam of news API calls."""
        return int(self.manager.get('trading.thesis_revalidation_interval_min', 15))

    @property
    def THESIS_REVALIDATION_LIMBO_R(self) -> float:
        """Re-validation only runs when |unrealized R| < this. Default 0.5.
        Outside ±0.5R, existing logic (breakeven_stop, proactive_exit,
        trailing stop, hard stop) handles the position. The re-check is
        for the in-between zone where the trade is sitting flat-ish and
        the underlying thesis may be silently breaking."""
        return float(self.manager.get('trading.thesis_revalidation_limbo_r', 0.5))

    @property
    def BREAKEVEN_ACTIVATION_R(self) -> float:
        """v-breakeven-stop-2026-04-28: how far in favor (in R-multiples,
        where R = entry-stop distance) a trade must move before its stop
        is lifted to entry. Default 0.5 = once trade is halfway to take-
        profit, lock in 'cannot lose' downside. Set 0 to disable.

        Live evidence 2026-04-28: PLTR long and TSLA long both went
        positive shortly after entry, then reversed to -0.94% / -0.80%
        before proactive_macd exit fired. Both could have exited flat
        if a breakeven stop had been armed at +0.5R. Net would have been
        $0 instead of -$231 across the two trades.

        Counter-risk: in chop, trades that briefly hit +0.5R then drift
        back to entry get stopped flat instead of getting another chance
        to run. Net positive because preventing winner→loser flips
        compounds, while flat exits cost only commissions."""
        return float(self.manager.get('trading.breakeven_activation_r', 0.5))

    @property
    def ENABLE_PROACTIVE_EXIT(self) -> bool:
        """v-proactive-exit-2026-04-27: when at -0.5R or worse AND a primary
        indicator (MACD / RSI / ADX) has flipped against the position, exit
        early instead of waiting for the full -1.0R stop.

        Live evidence 2026-04-27: RIOT short took the full -$236 stop after
        75 minutes; thesis was already broken at the -0.5R mark per indicator
        readings, but the bot rode the full move down because no proactive
        exit existed. Default True. Disable to restore "ride to stop" behavior.

        Counter-risk: catches some losses that would have reverted to wins
        (false breakouts of indicators near 50/MACD-cross). Net should be
        positive because we're already halfway to stop when this fires —
        the trade was already on a losing path."""
        return bool(self.manager.get('trading.enable_proactive_exit', True))

    @property
    def TRAIL_ACTIVATION_ATR_MULT(self) -> float:
        """ATR multiples the price must move in favor before the trailing
        stop engages. Default 2.0. Was hardcoded 1.0 until 2026-04-23 —
        too tight: trail fired at breakeven levels, locking in pennies.

        Evidence (2026-04-23, 14 trades): avg win +$16.58 vs avg loss
        -$56.98 = R:R 0.29, unprofitable even at 50% win rate because
        winners never reached the 3×ATR take-profit. Raising activation
        to 2×ATR forces price to prove momentum before the trail bites.

        Tuning: higher = more patience (winners run longer, small wins
        become no-wins); lower = more protection (catches more wins but
        smaller). Sweet spot typically between 1.5 and 2.5 for 5-min bars."""
        return float(self.manager.get('trading.trail_activation_atr_mult', 2.0))

    @property
    def TRAIL_WIDTH_ATR_MULT(self) -> float:
        """Once activated, the trailing stop sits this many ATRs behind
        the running peak. Default 1.5. Was hardcoded 1.0 — equal to the
        average intraday noise, so any normal pullback triggered exit.

        With entry stop at 1.5×ATR and trail width at 1.5×ATR, winners
        and losers have symmetric risk — so the 3×ATR take-profit can
        actually be reached by surviving normal pullbacks."""
        return float(self.manager.get('trading.trail_width_atr_mult', 1.5))

    @property
    def SIZING_STRENGTH_WEIGHT(self) -> float:
        """Weight of signal.strength in the combined sizing quality score.

        Sizing uses a blended quality = strength_weight × signal.strength
                                      + (1 - strength_weight) × signal.confidence

        Default 0.5 = equal blend. Set 0.0 to size purely on confidence
        (pre-v-size-by-strength behavior); 1.0 to size purely on strength.

        Why both matter:
          confidence  — strategy's estimate of win probability (Kelly input)
          strength    — magnitude of the specific setup (e.g., sentiment
                        compound 0.35 vs 0.80; oversold by 2% vs 5%)
        A high-confidence weak signal and a low-confidence strong signal
        both deserve less than full size. Blending captures both."""
        return float(self.manager.get('trading.sizing_strength_weight', 0.5))

    @property
    def KELLY_FLOOR(self) -> float:
        """Minimum Kelly multiplier for position sizing. Default 0.50.
        Was 0.25 — quarter-sized any sub-0.60 confidence signal, making
        the bot fire trivially small trades that commissions + slippage
        eat alive. Raising to 0.50 doubles the floor; high-confidence
        trades (> 0.75) naturally get full-Kelly."""
        return float(self.manager.get('trading.kelly_floor', 0.50))

    @property
    def MAX_POSITION_VALUE_BP_PCT(self) -> float:
        """Per-position notional cap as a fraction of BUYING POWER (not
        equity). Default 0.15 = 15% of BP. With max_positions=5, this
        caps total utilization at 5 × 15% = 75% of BP — leaves headroom
        for volatility and concentration rules.

        On $71k buying power (observed with 2.6x margin on $27k equity):
          0.15 × 71k = $10,650 per position  (up from old fixed $10k cap)
          0.20 × 71k = $14,200 per position
          0.25 × 71k = $17,750 per position

        RISK per trade still comes from MAX_RISK_PER_TRADE × EQUITY,
        NOT buying power. Position size grows with margin, but maximum
        loss stays tied to what we can actually afford (equity).

        The legacy fixed MAX_POSITION_VALUE ($10k) still applies as an
        absolute ceiling — whichever is smaller wins. Set
        trading.max_position_value very high to remove that ceiling."""
        return float(self.manager.get('trading.max_position_value_bp_pct', 0.20))

    @property
    def ENABLE_MEAN_REV_SHORT(self) -> bool:
        """Enable the mean-reversion SHORT branch (RSI > 70 + above BB upper).

        Default False. Live evidence 2026-04-22: 5 closed short trades,
        2 wins / 3 losses, realized -$115.03, profit factor 0.25.
        60-day backtest showed PF 1.02 — marginal on a neutral tape and
        a loser on an uptrending one (RSI>70 is the norm, not a reversal
        signal, during a rally).

        Structural issue: no symmetric "rising peak" filter exists to
        match the long side's "falling knife" filter. Short fires on any
        overbought reading, including peaks in confirmed uptrends that
        just keep making higher highs.

        Flip True in Config.yaml to re-enable — ideally after adding the
        trend-context filter (require close<SMA50 AND MACD<signal before
        taking the short). Existing short positions remain under the bot's
        exit management when this flag is off; only NEW shorts are blocked."""
        return bool(self.manager.get('trading.enable_mean_rev_short', False))

    @property
    def ENABLE_BREAKOUT_LONG(self) -> bool:
        """Enable the breakout-long strategy (20-bar high + volume + trend).

        Default False. 60-day 5-min backtest (2026-04-20, 20 top-volume
        symbols) showed breakout-long at PF 0.79-0.83 even with strict
        gates — 50% of entries stop out, only 18-21% hit target. The
        strategy is structurally weak at intraday timeframes (false
        breakouts dominate). Kept in code for daily-bar backtests and
        optional re-enable. Flip to True in Config.yaml to reactivate."""
        return bool(self.manager.get('trading.enable_breakout_long', False))

    @property
    def ENABLE_MOMENTUM_LONG(self) -> bool:
        """Enable the momentum-long strategy (MACD bullish + RSI healthy +
        ADX trending).

        Default False. 60-day 5-min backtest showed momentum-long at
        PF 0.84-0.85 even with strict gates (close>SMA50, RSI 55-65).
        51% of entries stop out, only 16% hit target, 31-35% time out.
        MACD bullish crossovers on 5-min bars produce too many false
        positives. Kept in code for higher-timeframe backtests. Flip to
        True in Config.yaml to reactivate."""
        return bool(self.manager.get('trading.enable_momentum_long', False))

    @property
    def ENABLE_STRICT_LONG_GATES(self) -> bool:
        """Enable the v-profitability-pass-2026-04-20 tighter gates for
        breakout-long and momentum-long. Default True (new stricter
        behavior); set False to restore pre-2026-04-20 gates.

        Changes when True:
          Breakout long: requires close > high_20 * 1.003 (0.3% real break,
                         not tick), ADX > 25 (was 20), volume_ratio > 2.0
                         (was 1.5).
          Momentum long: requires close > SMA50 (trend confirmation),
                         RSI 55-65 (was 50-70).

        Motivation: 60-day backtest showed breakout-long PF 0.83 and
        momentum-long PF 0.85 (both losing). High stop rate (~50%) and
        timeout rate (~31%) pointed at false-breakout and weak-trend
        entries. Tighter gates cut trade count and raise win rate."""
        return bool(self.manager.get('trading.enable_strict_long_gates', True))

    @property
    def ENABLE_RELAXED_MEAN_REV_LONG(self) -> bool:
        """Enable the v-profitability-pass-2026-04-20 relaxed falling-knife
        filter for mean-reversion LONG. Default True.

        Old behavior: skip if (close<SMA50 AND MACD<signal). Filtered out
        too many oversold-bounce setups — backtest showed only 57 trades
        in 60 days × 20 symbols at PF 2.43.
        New behavior: skip only if (close<SMA50 AND MACD<signal AND
        close<low_5) — require very-recent weakness on top of the two old
        signals to confirm the knife is still falling. Expected to raise
        trade count ~3-5x while keeping PF >= 1.5."""
        return bool(self.manager.get('trading.enable_relaxed_mean_rev_long', True))

    @property
    def ENABLE_SHORT_MIRRORS(self) -> bool:
        """Enable the v-short-mirrors-2026-04-20 short branches:
        - Breakout strategy: breakdown short (close < 20-bar low + trend/volume gates)
        - Momentum strategy: bearish-momentum short (MACD bearish + RSI 30-50 + ADX>25 + below SMA50)

        Default False. 60-day 5-min backtest (2026-04-20, 20 top-volume symbols)
        showed both short branches at PF 0.69 — symmetric shorts fight the oversold
        bounce. Existing mean-reversion short-side (RSI>70 fade) is unchanged by
        this flag. Flip to True for paper testing after re-tuning gate parameters."""
        return bool(self.manager.get('trading.enable_short_mirrors', False))

    @property
    def TRADING_API_KEY(self) -> Optional[str]:
        """Bearer token that protects the REST and WebSocket API endpoints.

        Returns the value of the TRADING_API_KEY environment variable, or None
        when the variable is not set (auth disabled / dev mode).
        """
        return os.getenv("TRADING_API_KEY")

    # Order execution constants
    @property
    def LIMIT_ORDER_SLIPPAGE(self):
        """Slippage factor for limit orders (e.g., 0.001 = 0.1%)"""
        return self.manager.get('trading.limit_order_slippage', 0.001)

    @property
    def DEFAULT_STOP_LOSS_PCT(self):
        """Default stop loss percentage for positions without explicit stops"""
        return self.manager.get('trading.default_stop_loss_pct', 0.05)

    @property
    def DEFAULT_TAKE_PROFIT_PCT(self):
        """Default take profit percentage for positions without explicit targets"""
        return self.manager.get('trading.default_take_profit_pct', 0.10)

    @property
    def MAX_POSITION_VALUE_PCT(self):
        """Maximum position value as fraction of portfolio"""
        return self.manager.get('trading.max_position_value_pct', 0.25)

    @property
    def ATR_STOP_MULTIPLIER(self):
        """ATR multiplier for stop-loss distance (1.5 = stop at 1.5×ATR from entry)."""
        return self.manager.get('trading.atr_stop_multiplier', 1.5)

    @property
    def ATR_REWARD_RISK_RATIO(self):
        """R:R ratio for take-profit relative to stop distance (2.0 = 2:1 R:R)."""
        return self.manager.get('trading.atr_reward_risk_ratio', 2.0)

    @property
    def RISK_PER_TRADE_PCT(self):
        """Fraction of equity risked per trade for ATR-based sizing.

        Default raised 0.010 → 0.015 on 2026-04-24. Live evidence from
        2026-04-23 and -24: notional per trade was $8-9k (good, via margin
        sizing) but dollar_risk was only $130-140 (1% × equity × Kelly 0.5).
        Winners at 0.3-1% moves only returned $28-84; losers at 1.5-2%
        stops cost $130-170. Raising the risk budget 50% grows BOTH wins
        and losses proportionally — the absolute $ values finally matter
        (wins $42-125, losses $195-255 per trade).

        Tuning: 0.01 = conservative (original), 0.015 = moderate (new),
        0.02 = aggressive. Anything above 0.02 is a professional-trader
        territory where a bad streak can deplete equity fast."""
        return self.manager.get('trading.risk_per_trade_pct', 0.015)

# Initialize configuration
config = Config()
# ============================================================================
# LOGGING SETUP
# ============================================================================

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for file output"""

    def format(self, record):
        import json as _json
        log_entry = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry['exception'] = self.formatException(record.exc_info)
        # Include extra fields for trade events
        for key in ('symbol', 'action', 'signal_type', 'order_id', 'price',
                     'quantity', 'pnl', 'reason'):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return _json.dumps(log_entry)


def setup_logging():
    """Configure logging with JSON file output and human-readable console output"""
    logger = logging.getLogger('TradingBot')
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        str(Config().LOG_PATH),
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

logger = setup_logging()
