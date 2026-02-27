// TypeScript interfaces matching Python dataclasses in gap_fade_app.py lines 1316-1608

export interface GapFadeConfig {
  // Gap detection
  gap_threshold: number;
  max_gap_pct: number;
  vol_ratio_max: number;
  min_avg_volume: number;
  min_price: number;

  // Position sizing
  initial_capital: number;
  risk_pct: number;
  kelly_fraction: number;
  max_positions: number;
  thin_day_threshold: number;

  // Stops and targets
  stop_pct: number;
  partial_target_pct: number;
  partial_cover_frac: number;
  bounce_entry_pct: number;

  // Adaptive stops
  adaptive_stops: boolean;
  stop_gap_fraction: number;
  stop_min_pct: number;
  stop_max_pct: number;

  // Market regime filter
  regime_filter: boolean;
  regime_spy_gap_limit: number;
  regime_spy_block_pct: number;
  regime_vix_threshold: number;

  // Re-entry after stop-out
  reentry_enabled: boolean;
  reentry_cooldown_minutes: number;
  reentry_max_per_symbol: number;
  reentry_stop_pct: number;
  reentry_trigger_pct: number;

  // Gap-down fading (longs)
  trade_gap_downs: boolean;
  gap_down_threshold: number;
  gap_down_max_pct: number;
  gap_down_vol_ratio_max: number;

  // Entry cutoff
  entry_cutoff_hour: number;
  entry_cutoff_min: number;

  // Hold thresholds
  min_hold_minutes: number;
  min_profit_take_pct: number;
  min_gap_fill_pct: number;

  // Time exits
  time_exit_hour: number;
  time_exit_min: number;
  eod_exit_hour: number;
  eod_exit_min: number;

  // Circuit breakers
  daily_loss_limit: number;
  max_consec_losses: number;
  max_drawdown: number;

  // Scanner universe
  scan_universe: 'alpaca' | 'study' | 'custom';
  custom_symbols: string;

  // Position cap
  max_notional: number;

  // Backtest
  backtest_years: number;

  // Live order execution
  limit_orders_only: boolean;
  limit_offset_pct: number;

  // Realism adjustments
  slippage_pct: number;
  borrow_rate_annual: number;
  max_pct_adv: number;
  adverse_fill: boolean;
  adverse_fill_pct: number;

  // Catalyst detection
  catalyst_enabled: boolean;
  catalyst_skip_earnings: boolean;
  catalyst_earnings_penalty: number;
  catalyst_news_penalty: number;
  catalyst_noise_bonus: number;

  // Automation
  auto_start: boolean;

  // Telegram alerts
  alert_telegram_enabled: boolean;
  alert_telegram_token: string;
  alert_telegram_chat_id: string;

  // LLM Supervisor
  llm_enabled: boolean;
  llm_url: string;
  llm_model: string;
  llm_timeout: number;
  llm_max_failures: number;
  llm_circuit_reset: number;
  llm_max_hold_overrides: number;
}

export interface GapCandidate {
  symbol: string;
  gap_pct: number;
  prev_close: number;
  premarket_price: number;
  avg_vol_20d: number;
  vol_ratio: number;
  shortable: boolean;
  easy_to_borrow: boolean;
  score: number;
  catalyst: string;
  catalyst_detail: string;
  direction: 'short' | 'long';
}

export interface GapPosition {
  symbol: string;
  shares: number;
  entry_price: number;
  stop_price: number;
  half_target: number;
  full_target: number;
  prev_close: number;
  partial_filled: boolean;
  entry_time: string;
  remaining_shares: number;
  closing: boolean;
  entry_order_id: string;
  entry_fill_price: number;
  stop_order_id: string;
  direction: 'short' | 'long';
  llm_hold_overrides: number;
  high_water_pnl_pct: number;
  high_water_price: number;
  high_water_time: string;
  partial_fill_time: string;
  last_prices: string;
  last_llm_profit_check: number;
  gap_pct: number;
  vol_ratio: number;
  score: number;
  catalyst: string;
}

export interface TradeRecord {
  symbol: string;
  entry_price: number;
  exit_price: number;
  shares: number;
  pnl: number;
  pnl_pct: number;
  entry_time: string;
  exit_time: string;
  exit_reason: 'stop' | 'partial' | 'full_target' | 'time_exit' | 'eod' | 'manual';
  holding_minutes: number;
  side: 'short' | 'long';
  gap_pct: number;
  vol_ratio: number;
  score: number;
  catalyst: string;
}

export interface DailyStats {
  date: string;
  trades: number;
  wins: number;
  losses: number;
  pnl: number;
  peak_equity: number;
  consecutive_losses: number;
  halted: boolean;
  halt_reason: string;
}

export interface StopOutRecord {
  symbol: string;
  stop_time: string;
  original_entry: number;
  prev_close: number;
  gap_pct: number;
  avg_vol_20d: number;
  reentry_count: number;
  direction: 'short' | 'long';
}

export interface MarketRegime {
  spy_gap_pct: number;
  vix_level: number;
  position_reduction: number;
  note: string;
}

// API response from GET /api/state
export interface TradingState {
  status: TraderStatus;
  equity: number;
  peak_equity: number;
  positions: Record<string, GapPosition>;
  daily_stats: DailyStats;
  candidates: GapCandidate[];
  last_scan_time: string;
  metrics: Metrics;
  messages: Message[];
  config: GapFadeConfig;
  today_trades: TradeRecord[];
  llm: LLMStatus;
}

export type TraderStatus =
  | 'stopped'
  | 'trading'
  | 'paused'
  | 'scanning'
  | 'waiting'
  | 'standdown'
  | 'halted';

export interface Metrics {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  total_pnl: number;
  avg_pnl: number;
  avg_winner: number;
  avg_loser: number;
  best_trade: number;
  worst_trade: number;
  sharpe: number;
  max_drawdown: number;
  avg_holding_minutes: number;
}

export interface Message {
  time: string;
  text: string;
  level?: 'info' | 'warning' | 'error' | 'success';
}

export interface LLMStatus {
  enabled: boolean;
  available: boolean;
  model: string;
  circuit_open: boolean;
  failure_count: number;
}

// WebSocket message types
export type WSMessageType =
  | 'live_status'
  | 'trade'
  | 'positions_update'
  | 'scan_results'
  | 'tracker_tick'
  | 'backtest_progress'
  | 'backtest_complete'
  | 'bt_log'
  | 'db_build_progress';

export interface WSMessage {
  type: WSMessageType;
  [key: string]: unknown;
}

export interface WSLiveStatus {
  type: 'live_status';
  status: TraderStatus;
}

export interface WSTrade {
  type: 'trade';
  trades: TradeRecord[];
}

export interface WSTrackerTick {
  type: 'tracker_tick';
  symbol: string;
  price: number;
  bid: number;
  ask: number;
}

export interface WSBacktestProgress {
  type: 'backtest_progress';
  pct: number;
  message: string;
}

export interface WSBacktestComplete {
  type: 'backtest_complete';
  result: BacktestResult;
}

export interface BacktestResult {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  total_pnl: number;
  sharpe: number;
  max_drawdown: number;
  equity_curve: number[];
  trades: TradeRecord[];
}

export interface HealthResponse {
  status: string;
  uptime: number;
  positions: number;
  trader_status: TraderStatus;
}

export interface AccountInfo {
  equity: number;
  buying_power: number;
  cash: number;
  portfolio_value: number;
  pattern_day_trader: boolean;
}

export interface TrackerBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface DBStats {
  total_symbols: number;
  total_bars: number;
  date_range: string;
  last_updated: string;
}
