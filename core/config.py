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
        # v-max-positions-default-2026-05-06: 5 → 10. Hard cap on
        # concurrent bot-managed trades. Positions that are hands-off
        # (managed_by_bot=False, e.g. pre-existing Schwab holdings) do
        # NOT count against this limit.
        return self.manager.get('trading.max_positions', 10)

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
    def ENABLE_MEAN_REV_UPTREND_PULLBACK(self) -> bool:
        """v-mean-rev-uptrend-pullback-2026-05-28: alternative mean-rev
        entry for stocks pulling back to support inside an uptrend.

        Operator instruction 2026-05-28 ~10:00 ET after 3 sessions of
        the bot being silent while specific names ran on momentum.
        Existing mean-rev fires only on extreme oversold (RSI<30 +
        close below BB lower); never catches the "in uptrend, pulling
        back to support" pattern that's been today's regime.

        New trigger:
          RSI between 30 and 45 (mild)
          close >= SMA50 (uptrend confirmation)
          close within 2% of BB lower OR up to 1% below it
        Same downstream gates: price-direction (green bar + vol>=1.5x
        + rejection of lows). Same ATR-based stop/target. Pattern is
        logged as `signal_buy reason=uptrend_pullback` for audit.

        Default True. Toggle False via API if it produces too many
        false starts. Reversible without code change.
        """
        return bool(
            self.manager.get(
                'trading.enable_mean_rev_uptrend_pullback', True
            )
        )

    @property
    def ENABLE_NEWS_TECHNICAL_CONFIRMATION(self) -> bool:
        """v-news-confirmation-gate-2026-05-27: require technical
        confirmation before news strategy fires signal_buy/signal_sell.

        Operator decision 2026-05-27 after 3-for-3 losing entries
        across 2 sessions (FLY/ASTS 2026-05-26, TSLA would-have on
        2026-05-27) all of shape: bullish news sentiment + already-
        moved price. News alone is a lagging indicator. With this
        gate on, news_strategy demands matching technical state:
          BUY: rsi<55, price within 5% of sma_20, macd_val>=macd_sig,
               volume_ratio>=1.2x average. SELL: inverted.

        Default True. Toggle False via API
        (PUT /api/settings/trading) if the gate proves too restrictive
        in practice and we lose real opportunities. Audit grep:
        `engine_decision .* reason=no_technical_confirmation`.
        """
        return bool(
            self.manager.get(
                'trading.enable_news_technical_confirmation', True
            )
        )

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
    def NEWS_VERIFIER_ADVISORY(self) -> bool:
        """v-news-verifier-advisory-2026-05-08: run the fresh-news
        verifier in advisory mode — it executes and logs its verdict
        on every news signal, but does NOT veto the trade.

        Rationale: the operator complained that 100% gating felt opaque
        (you can't tell whether a veto was right without seeing the
        outcome). Advisory mode emits an audit line
        `engine_decision component=news_verifier_advisory action=advisory
        reason=<verdict> fresh_count=N latest_age_min=...` for every
        news entry, AND records the decision against the eventual
        bot_trades outcome. After a week of data we can grade whether
        the verifier's vetoes would have improved or hurt P&L, then
        flip ENABLE_NEWS_VERIFIER True with evidence.

        Default True — the cost of running verification is one HTTP
        call; the benefit is observability. Set False to silence."""
        return bool(self.manager.get('trading.news_verifier_advisory', True))

    # ── v-side-classifier-config-2026-05-13 ───────────────────────────
    # Side-classifier subsystem flags. All default OFF so the live
    # trading path is unaffected until the operator explicitly opts in.
    # Rollout flavor B (per-strategy gradual) chosen 2026-05-13:
    # USE_SIDE_CLASSIFIER is the master switch; the per-strategy
    # switches let us gate news_strategy first, then mean_reversion.

    @property
    def USE_SIDE_CLASSIFIER(self) -> bool:
        """Master switch for the side classifier. Default False.
        When False, no part of `core/classifier` is consulted by the
        live trading path. Flip via `Config().yaml: trading.use_side_classifier`."""
        return bool(self.manager.get('trading.use_side_classifier', False))

    @property
    def SIDE_CLASSIFIER_NEWS_STRATEGY(self) -> bool:
        """Per-strategy gate. When `USE_SIDE_CLASSIFIER=True`, this
        decides whether news_strategy signals pass through the
        classifier. Rollout flavor B starts with this True, then
        flips mean-reversion on after N clean sessions."""
        return bool(self.manager.get('trading.side_classifier_news_strategy', False))

    @property
    def SIDE_CLASSIFIER_MEAN_REVERSION(self) -> bool:
        """Per-strategy gate for mean-reversion entries. Starts False
        and flips True after news_strategy gating proves itself."""
        return bool(self.manager.get('trading.side_classifier_mean_reversion', False))

    @property
    def SIDE_CLASSIFIER_LONG_THRESHOLD(self) -> float:
        """Minimum `long_score` for the rule layer to allow LONG.
        Default 0.55 per PLAN §5.3. Tunable post-backtest."""
        return float(self.manager.get('trading.side_classifier_long_threshold', 0.55))

    @property
    def SIDE_CLASSIFIER_SHORT_THRESHOLD(self) -> float:
        """Minimum `short_score` for the rule layer to allow SHORT.
        Default 0.55 per PLAN §5.3."""
        return float(self.manager.get('trading.side_classifier_short_threshold', 0.55))

    @property
    def SIDE_CLASSIFIER_CACHE_TTL_SEC(self) -> int:
        """How long a per-symbol classifier decision stays valid before
        re-computation. Default 1800s (30min) per PLAN §7."""
        return int(self.manager.get('trading.side_classifier_cache_ttl_sec', 1800))

    @property
    def SIDE_CLASSIFIER_SHADOW_MODE(self) -> bool:
        """Run the classifier alongside live trading but log only.
        Decisions are surfaced in audit logs (component=side_classifier)
        without affecting any gating. Default False.

        v-classifier-runtime-2026-05-13. Per ``.claude/PLAN_trained_side_
        classifier.md`` §8: shadow → advisory → live. This flag is
        what 'shadow' looks like at the code level.
        """
        return bool(self.manager.get(
            'trading.side_classifier_shadow_mode', False))

    @property
    def SIDE_CLASSIFIER_MODEL_PATH(self) -> str:
        """Filesystem path to the FFN (or future TFT) checkpoint.
        Defaults to the watchlist-v1 smoke checkpoint we trained today.
        If the file is absent, ClassifierRuntime fails open
        (rule-only decisions) and logs a single WARNING."""
        return str(self.manager.get(
            'trading.side_classifier_model_path',
            'models/classifier/ffn_watchlist_v1.pt'))

    @property
    def SCREENER_LOOP_SEC(self) -> int:
        """v-parallel-screener-loop-2026-05-12: cadence for the supervised
        screener loop. Default 120s matches the old in-line gate inside
        `_analyze_markets_with_commentary`. Tuning down increases Yahoo +
        Schwab REST traffic; tuning up means a slower watchlist refresh
        when the tape rotates."""
        return int(self.manager.get('trading.screener_loop_sec', 120))

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
    def PROACTIVE_EXIT_MIN_AGE_NEWS(self) -> int:
        """v-proactive-time-floor-2026-04-30: minimum minutes a news-driven
        position must be open before proactive_exit can fire. Whipsaw
        analysis on 2026-04-30 showed 5 of 5 whipsaws closed within 5 min
        of entry; news theses (institutional re-rate after earnings,
        guidance, M&A) play out over hours, not 5-min bars. Holding
        through the first 30 min would have saved ~$1,400 today.
        Default 30. Lower to 15 for faster news plays; raise to 60 if
        whipsaw rate is still high after measurement."""
        return int(self.manager.get('trading.proactive_exit_min_age_news', 30))

    @property
    def PROACTIVE_EXIT_MIN_AGE_MEANREV(self) -> int:
        """Min age before proactive_exit fires for mean-reversion trades.
        Bounces are typically fast — 10 min gives the snap-back time to
        materialize without letting MACD-flip wiggle kick us out
        prematurely. Default 10."""
        return int(self.manager.get('trading.proactive_exit_min_age_meanrev', 10))

    @property
    def PROACTIVE_EXIT_MIN_AGE_DEFAULT(self) -> int:
        """Min age for proactive_exit on any other strategy
        (momentum, breakout, ML, etc). Default 15 — middle ground."""
        return int(self.manager.get('trading.proactive_exit_min_age_default', 15))

    @property
    def LIVE_SIZE_MULTIPLIER(self) -> float:
        """Global live-launch safety dial applied after strategy multipliers.

        Composes with ``strategy_size_multipliers`` — final sizing is
        ``base * strategy_mult * live_mult``. Default 1.0 (no change).
        Set to 0.25 for the 2026-05-26 quarter-size live launch; graduate
        when walk-forward bootstrap CI lower bound clears 1.0 (use
        ``research/news_strategy_validation.py`` to measure).
        """
        try:
            return float(self.manager.get('trading.live_size_multiplier', 1.0))
        except (TypeError, ValueError):
            return 1.0

    @property
    def LATE_ENTRY_CUTOFF_HOUR(self) -> int:
        """Hour-of-day ET past which no new entries are accepted. See
        engine.py late-entry cutoff guard. Default 15 (3 PM ET)."""
        return int(self.manager.get('trading.late_entry_cutoff_hour', 15))

    @property
    def LATE_ENTRY_CUTOFF_MINUTE(self) -> int:
        """Minute-of-hour ET past which no new entries are accepted on the
        cutoff hour. Default 30 — combined with hour=15, gives 15:30 ET."""
        return int(self.manager.get('trading.late_entry_cutoff_minute', 30))

    def STRATEGY_SIZE_MULTIPLIER(self, strategy_name: str) -> float:
        """Per-strategy size multiplier applied after Kelly sizing.

        Looked up by ``signal.reasoning['strategy']`` in
        ``risk/manager.py``. Missing or invalid keys return 1.0
        (no change).

        Configured via ``trading.strategy_size_multipliers`` (dict).
        Used to under-size strategies still proving edge — e.g.
        mean_reversion at 0.5 until N>=20 confirms PF>=1.5.
        """
        mults = self.manager.get('trading.strategy_size_multipliers', {}) or {}
        if not isinstance(mults, dict):
            return 1.0
        try:
            return float(mults.get(strategy_name, 1.0))
        except (TypeError, ValueError):
            return 1.0

    # ──────────────────────────────────────────────────────────────────
    # v-loop-decoupling-2026-04-30 (Phase 2): cadence + staleness knobs
    # for the three-task model (analysis_loop / position_loop / streamer).
    # ──────────────────────────────────────────────────────────────────
    @property
    def POSITION_LOOP_SEC(self) -> float:
        """Cadence of the FSM exit dispatcher loop. 1.0s = stops fire
        within ~1s of breach. Lowering helps responsiveness; raising
        saves CPU. Below 0.5s noisy, above 3s defeats the purpose."""
        return float(self.manager.get('trading.position_loop_sec', 1.0))

    @property
    def QUOTE_REFRESH_SEC(self) -> float:
        """Cadence of the quote streamer per active symbol. 3s × 10
        symbols = ~3.3 req/s to Schwab. Tighten only if you have
        broker headroom."""
        return float(self.manager.get('trading.quote_refresh_sec', 3.0))

    @property
    def ANALYSIS_LOOP_SEC(self) -> float:
        """Cadence of the slow loop (screener / signal routing / new
        entries). 60s aligns with 1-min bars; finer cadence buys
        nothing for a 5-min-bar bot."""
        return float(self.manager.get('trading.analysis_loop_sec', 60.0))

    @property
    def ANALYSIS_LOOP_RTH_SEC(self) -> float:
        """v-off-hours-analysis-2026-05-01: cadence DURING regular
        trading hours. 30s matches the legacy default; the analysis
        loop runs the full screener/signal/ML pipeline this often.
        Tighten only if you have Schwab quote-budget headroom."""
        return float(self.manager.get('trading.analysis_loop_rth_sec', 30.0))

    @property
    def ANALYSIS_LOOP_OFF_HOURS_SEC(self) -> float:
        """v-off-hours-analysis-2026-05-01: cadence OUTSIDE regular
        hours. Analysis still runs (news, indicators, ML) but at a
        reduced rate — data sources don't update meaningfully overnight
        and the screener/quotes burn budget for no return. Default 300s
        (5 min) keeps the bot informed without hammering APIs.
        Signal-router market_hours gate still blocks new orders."""
        return float(self.manager.get('trading.analysis_loop_off_hours_sec', 300.0))

    @property
    def QUOTE_MAX_STALE_SEC(self) -> float:
        """Soft cliff: position_loop will SKIP exit evaluation on a
        quote older than this. 15s tolerates a Schwab hiccup but never
        acts on yesterday's price."""
        return float(self.manager.get('trading.quote_max_stale_sec', 15.0))

    @property
    def QUOTE_HARD_STALE_SEC(self) -> float:
        """Hard cliff: cache older than this means the streamer is
        broken. Trips an alert; new entries pause via analysis_loop
        gating. Default 60s."""
        return float(self.manager.get('trading.quote_hard_stale_sec', 60.0))

    @property
    def QUOTE_FETCH_TIMEOUT_SEC(self) -> float:
        """Per-symbol quote fetch timeout. 3s gives Schwab room; longer
        starves other symbols when one hangs."""
        return float(self.manager.get('trading.quote_fetch_timeout_sec', 3.0))

    @property
    def QUOTE_FETCH_CONCURRENCY(self) -> int:
        """Max in-flight quote fetches at once across all symbols."""
        return int(self.manager.get('trading.quote_fetch_concurrency', 8))

    @property
    def TASK_RESTART_MAX_CRASHES(self) -> int:
        """Supervisor: number of crashes-in-window before circuit-break."""
        return int(self.manager.get('trading.task_restart_max_crashes', 3))

    @property
    def TASK_RESTART_WINDOW_SEC(self) -> float:
        """Supervisor: rolling window for crash counting."""
        return float(self.manager.get('trading.task_restart_window_sec', 300.0))

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
    def ENABLE_UNIVERSE_QUALITY_FILTER(self) -> bool:
        """v-universe-quality-filter-2026-05-22: filter screener candidates
        at universe-build time before strategies see them. Three sub-gates:
        leveraged-ETF blocklist, close > SMA50 (uptrend), and 5-day return
        > SPY 5-day return (relative strength). Default True — turn off
        in yaml to backtest the impact (set `trading.enable_universe_quality_filter: false`)."""
        return bool(self.manager.get('trading.enable_universe_quality_filter', True))

    @property
    def USE_OVERSOLD_BOUNCE_V2(self) -> bool:
        """v-oversold-v2-2026-05-19: gate for the OversoldBounceV2Strategy.
        Default changed to FALSE on 2026-05-19 after backtest showed v2
        underperforms legacy (12 trades, 16.7% win rate, PF 1.04 vs legacy's
        62 trades, 61.3% win rate, PF 1.72 on 30-day NVDA/AMD/RKLB/FCEL/POET
        sample). Strategy code retained for further iteration. Flip True
        in yaml only after v2 outperforms in a fresh backtest."""
        return bool(self.manager.get('trading.use_oversold_bounce_v2', False))

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
    def MEAN_REV_ATR_STOP_MULTIPLIER(self):
        """v-mean-rev-wider-stop-2026-05-20: mean-reversion-specific stop
        multiplier. Default 2.5× ATR — wider than the global 1.5×.

        Rationale: with the bounce-confirmation filter (v-bounce-2026-05-20)
        only firing on real reversal candles, mean-rev entries are higher
        quality and deserve room to breathe through normal intraday
        whipsaw. 1.5× ATR puts stops inside the noise band on volatile
        names (FIG on 5/20: stopped at $22.16, day low $21.78, current
        $22.23 — recovered above the stop level). Wider stop = fewer
        whipsaw exits but bigger loss when stop genuinely hits;
        position-sizing automatically scales down to keep dollar_risk
        constant per trade.

        Falls back to ATR_STOP_MULTIPLIER if yaml key missing."""
        v = self.manager.get('trading.mean_rev_atr_stop_multiplier', None)
        if v is None:
            return 2.5  # mean-rev-specific default
        return v

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
