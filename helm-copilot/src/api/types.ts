/**
 * API Types for Helm Co-Pilot
 * 
 * Response shapes derived from the trading bot's FastAPI routes.
 * All types are defensive — nullable fields and optional properties
 * reflect the reality of a live trading system.
 */

// ============================================================================
// Status & System
// ============================================================================

export interface BotStatus {
  status: 'success' | 'error'
  bot_running: boolean
  mode: 'live' | 'simulation' | 'not_started' | string
  positions_count: number
  schwab_connected: boolean
  uptime: string
}

export interface SystemStats {
  status: 'success' | 'error'
  profile: {
    name: string
    confirmation_timeout_sec: number
    confirmation_timeout_action: string
    require_close_confirmation: boolean
    flatten_on_circuit: boolean
  }
  news_bus: NewsBusStats | null
  news_loop: NewsLoopStatus | null
  supervisor: SupervisorStatus | null
  last_wake_reason: string | null
}

export interface NewsBusStats {
  total_items: number
  symbols_tracked: number
  high_impact_items: number
  refresh_count: number
  items_evicted: number
  fetch_errors: number
  wake_events_fired: number
  last_refresh: string | null
  gate_pass: number
  gate_veto_stale: number
  gate_veto_low_tier: number
  gate_veto_no_corroboration: number
  gate_size_reduced: number
}

export interface NewsLoopStatus {
  running: boolean
  last_fetch: string | null
  errors: number
}

export interface SupervisorStatus {
  tasks: Record<string, string>
  crash_counts: Record<string, number>
}

// ============================================================================
// Market Indices
// ============================================================================

export interface MarketIndex {
  symbol: string
  name: string
  last: number
  change_pct: number
}

export interface MarketIndicesResponse {
  indices: MarketIndex[]
  last_updated: string | null
  stale: boolean
  error: string | null
}

// ============================================================================
// Account
// ============================================================================

export interface AccountStats {
  status: 'success' | 'error'
  source: 'schwab' | 'simulation' | 'internal' | 'none'
  stats: {
    balance: number
    buying_power: number
    daily_pnl: number
    total_pnl: number
    position_count: number
    cash: number
  }
}

// ============================================================================
// Positions
// ============================================================================

export interface Position {
  symbol: string
  side: 'long' | 'short'
  strategy: string | null
  mode: 'live' | 'simulation' | string
  entry_time: string | null
  entry_price: number
  current_price: number
  quantity: number
  stop_loss: number
  take_profit: number
  trailing_stop: number | null
  scaled_out: boolean
  unrealized_pnl: number
  updated_at: string
  managed_by_bot: boolean
  is_stale?: boolean
  is_long_term?: boolean
  day_pnl?: number
  pnl_percent?: number
  market_value?: number
}

export interface PositionsResponse {
  status: 'success' | 'error'
  count: number
  positions: Position[]
  message?: string
}

// ============================================================================
// Decisions
// ============================================================================

export interface Decision {
  id: number
  ts: string
  component: string
  symbol: string
  action: 'signal_buy' | 'signal_sell' | 'skip' | 'veto' | 'error' | string
  reason: string
  confidence: number | null
  meta_proba: number | null
  atr: number | null
  price: number | null
  details_json: Record<string, unknown> | null
}

export interface DecisionsResponse {
  status: 'success' | 'error'
  count: number
  decisions: Decision[]
  message?: string
}

// ============================================================================
// Decision Snapshots (for ML/detailed view)
// ============================================================================

export interface PriceVolumeFeatures {
  price: number
  returns_1: number
  returns_5: number
  returns_20: number
  price_vs_sma20: number
  price_vs_sma50: number
  rsi: number
  macd: number
  macd_signal: number
  macd_hist: number
  bb_position: number
  bb_width: number
  volume_ratio: number
  volume_std_ratio: number
  volume_trend: number
  atr_ratio: number
  high_low_ratio: number
  std_dev_ratio: number
}

export interface NewsAggregate {
  article_count: number
  fresh_count: number
  avg_sentiment: number
  freshest_age_sec: number | null
  source_tier_min: number | null
  high_impact_count: number
  corroboration_n: number
  gate_action: string
  gate_size_mult: number
}

export interface RegimeContext {
  regime: string
  spy_slope_pct: number
  vix: number | null
  tape: string | null
  er: number | null
}

export interface DecisionSnapshot {
  snapshot_id: string
  symbol: string
  ts: string
  mode: string
  strategy_id: string
  action: 'signal_buy' | 'signal_sell' | 'skip' | 'veto' | 'error'
  reason: string
  gate_name: string | null
  confidence: number
  price_vol: PriceVolumeFeatures
  news: NewsAggregate
  regime: RegimeContext
  momentum: MomentumContext | null
  would_entry_price: number | null
  would_stop_loss: number | null
  would_take_profit: number | null
  would_size_shares: number | null
  would_size_mult: number | null
  extra: Record<string, unknown>
}

export interface MomentumContext {
  session_id: string
  setup_type: string
  risk_off: boolean
  mc_size_mult: number
  rs_vs_spy: number
  is_day_trade: boolean
  flatten_hour: number
  entry_pattern: string
}

export interface DecisionSnapshotsResponse {
  status: 'success' | 'error'
  count: number
  snapshots: DecisionSnapshot[]
  message?: string
}

// ============================================================================
// News Bus
// ============================================================================

export interface NewsItem {
  id: string
  symbol: string
  headline: string
  summary: string | null
  source: string
  source_tier: number
  sentiment: number
  is_high_impact: boolean
  published_at: string
  fetched_at: string
  age_sec: number
}

export interface NewsBusResponse {
  status: 'success' | 'error'
  symbol: string
  as_of: string
  item_count: number
  aggregate: {
    article_count: number
    avg_sentiment: number
    high_impact_count: number
  } | null
  items: NewsItem[]
  has_high_impact: boolean
  message?: string
}

// ============================================================================
// WebSocket Messages
// ============================================================================

export interface WsDashboardUpdate {
  type: 'dashboard_update'
  data: {
    account: {
      balance: number
      buying_power: number
      daily_pnl: number
      margin_call: boolean
      cash: number
    }
    simulated_positions: WsPosition[]
    real_positions: WsPosition[]
    trades: WsTrade[]
    screener: WsScreenerItem[]
    live_quotes: Record<string, WsLiveQuote>
    symbol_intel?: WsSymbolIntel[]
  }
}

export interface WsPosition {
  symbol: string
  quantity: number
  entry_price: number
  current_price: number
  stop_loss: number | null
  take_profit: number | null
  unrealized_pnl: number
  is_stale?: boolean
  type: 'simulated' | 'real'
  mode: string
  managed_by_bot: boolean
  is_long_term?: boolean
  day_pnl?: number
  pnl_percent?: number
  market_value?: number
}

export interface WsTrade {
  symbol: string
  side: 'buy' | 'sell'
  quantity: number
  price: number
  time: string
}

export interface WsScreenerItem {
  symbol: string
  last: number
  change: number
  volume: number
  volatility: number
  high: number
  low: number
}

export interface WsLiveQuote {
  price: number | null
  bid: number | null
  ask: number | null
  is_stale: boolean
  source: string | null
  age_sec: number | null
}

export interface WsSymbolIntel {
  symbol: string
  quality: number
  signals: string[]
}

export interface WsSentimentUpdate {
  type: 'sentiment_update'
  data: {
    market_sentiment: number
    symbols: Record<string, number>
    alerts: WsSentimentAlert[]
  }
}

export interface WsSentimentAlert {
  id: string
  symbol: string
  message: string
  urgency: 'critical' | 'high' | 'medium' | 'low'
  timestamp: string
}

export interface WsCommentary {
  type: 'commentary'
  data: {
    symbol: string
    message: string
    timestamp: string
  }
}

export type WsMessage = WsDashboardUpdate | WsSentimentUpdate | WsCommentary

// ============================================================================
// Connection State
// ============================================================================

export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'offline' | 'demo'

export interface ConnectionInfo {
  state: ConnectionState
  lastConnected: Date | null
  lastError: string | null
  isDemo: boolean
}
