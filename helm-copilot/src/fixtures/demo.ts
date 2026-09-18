/**
 * Demo Fixtures for Helm Co-Pilot
 * 
 * Used when API is unreachable. ALWAYS labeled as DEMO — never as live data.
 * These fixtures demonstrate the UI capabilities without misleading users.
 */

import type {
  BotStatus,
  SystemStats,
  MarketIndicesResponse,
  AccountStats,
  PositionsResponse,
  DecisionSnapshot,
  DecisionSnapshotsResponse,
  NewsBusResponse,
  WsDashboardUpdate,
} from '../api/types'

// ============================================================================
// Demo Decision Cards (2-3 as requested)
// ============================================================================

export const DEMO_DECISION_CARDS: DecisionSnapshot[] = [
  {
    snapshot_id: 'demo-001',
    symbol: 'NVDA',
    ts: new Date(Date.now() - 180000).toISOString(), // 3 min ago
    mode: 'demo',
    strategy_id: 'momentum_breakout',
    action: 'signal_buy',
    reason: 'breakout_confirmed',
    gate_name: null,
    confidence: 0.78,
    price_vol: {
      price: 892.45,
      returns_1: 0.012,
      returns_5: 0.034,
      returns_20: 0.089,
      price_vs_sma20: 1.024,
      price_vs_sma50: 1.056,
      rsi: 0.62,
      macd: 0.0023,
      macd_signal: 0.0018,
      macd_hist: 0.0005,
      bb_position: 0.72,
      bb_width: 0.045,
      volume_ratio: 1.8,
      volume_std_ratio: 1.2,
      volume_trend: 0.15,
      atr_ratio: 0.028,
      high_low_ratio: 0.022,
      std_dev_ratio: 0.031,
    },
    news: {
      article_count: 4,
      fresh_count: 2,
      avg_sentiment: 0.65,
      freshest_age_sec: 1200,
      source_tier_min: 1,
      high_impact_count: 1,
      corroboration_n: 3,
      gate_action: 'full_size',
      gate_size_mult: 1.0,
    },
    regime: {
      regime: 'trending_bull',
      spy_slope_pct: 0.15,
      vix: 14.2,
      tape: 'risk_on',
      er: 0.68,
    },
    momentum: {
      session_id: new Date().toISOString().split('T')[0],
      setup_type: 'momentum_breakout',
      risk_off: false,
      mc_size_mult: 1.0,
      rs_vs_spy: 1.8,
      is_day_trade: true,
      flatten_hour: 15,
      entry_pattern: 'breakout',
    },
    would_entry_price: 892.45,
    would_stop_loss: 878.50,
    would_take_profit: 920.00,
    would_size_shares: 50,
    would_size_mult: 1.0,
    extra: {
      thesis: 'NVDA breaking out of consolidation on strong AI chip demand. Volume 1.8x average confirms institutional participation. RSI healthy at 62, not overbought. News catalyst: partnership announcement with major cloud provider.',
      risks: [
        'Semiconductor sector rotation risk if broader market sells off',
        'Valuation stretched at 35x forward P/E',
        'Previous resistance at $900 may cause hesitation'
      ],
      sources: [
        { name: 'Reuters', tier: 1, headline: 'NVIDIA announces expanded cloud partnership' },
        { name: 'Bloomberg', tier: 1, headline: 'Chip stocks rally on AI infrastructure spending' },
        { name: 'CNBC', tier: 2, headline: 'Analysts raise NVDA price targets' }
      ]
    },
  },
  {
    snapshot_id: 'demo-002',
    symbol: 'AAPL',
    ts: new Date(Date.now() - 600000).toISOString(), // 10 min ago
    mode: 'demo',
    strategy_id: 'mean_reversion',
    action: 'skip',
    reason: 'low_confidence',
    gate_name: 'news_gate',
    confidence: 0.42,
    price_vol: {
      price: 178.32,
      returns_1: -0.008,
      returns_5: -0.015,
      returns_20: 0.012,
      price_vs_sma20: 0.992,
      price_vs_sma50: 1.008,
      rsi: 0.45,
      macd: -0.0008,
      macd_signal: 0.0002,
      macd_hist: -0.001,
      bb_position: 0.38,
      bb_width: 0.032,
      volume_ratio: 0.85,
      volume_std_ratio: 0.6,
      volume_trend: -0.08,
      atr_ratio: 0.018,
      high_low_ratio: 0.015,
      std_dev_ratio: 0.022,
    },
    news: {
      article_count: 2,
      fresh_count: 0,
      avg_sentiment: 0.1,
      freshest_age_sec: 7200,
      source_tier_min: 2,
      high_impact_count: 0,
      corroboration_n: 1,
      gate_action: 'veto_stale',
      gate_size_mult: 0.0,
    },
    regime: {
      regime: 'choppy',
      spy_slope_pct: 0.02,
      vix: 16.8,
      tape: 'neutral',
      er: 0.35,
    },
    momentum: null,
    would_entry_price: null,
    would_stop_loss: null,
    would_take_profit: null,
    would_size_shares: null,
    would_size_mult: null,
    extra: {
      thesis: 'Mean reversion setup near SMA20, but signal lacks conviction. Low volume and stale news suggest waiting for clearer catalyst.',
      risks: [
        'No fresh news catalyst — stale information environment',
        'Low confidence score below threshold',
        'Market regime choppy — unfavorable for mean reversion'
      ],
      sources: []
    },
  },
  {
    snapshot_id: 'demo-003',
    symbol: 'TSLA',
    ts: new Date(Date.now() - 1800000).toISOString(), // 30 min ago
    mode: 'demo',
    strategy_id: 'correlation_guard',
    action: 'veto',
    reason: 'high_correlation_risk',
    gate_name: 'correlation_guard',
    confidence: 0.65,
    price_vol: {
      price: 245.80,
      returns_1: 0.025,
      returns_5: 0.048,
      returns_20: 0.092,
      price_vs_sma20: 1.045,
      price_vs_sma50: 1.082,
      rsi: 0.71,
      macd: 0.0042,
      macd_signal: 0.0035,
      macd_hist: 0.0007,
      bb_position: 0.85,
      bb_width: 0.068,
      volume_ratio: 2.1,
      volume_std_ratio: 1.8,
      volume_trend: 0.22,
      atr_ratio: 0.042,
      high_low_ratio: 0.038,
      std_dev_ratio: 0.052,
    },
    news: {
      article_count: 8,
      fresh_count: 5,
      avg_sentiment: 0.72,
      freshest_age_sec: 300,
      source_tier_min: 1,
      high_impact_count: 2,
      corroboration_n: 5,
      gate_action: 'full_size',
      gate_size_mult: 1.0,
    },
    regime: {
      regime: 'trending_bull',
      spy_slope_pct: 0.18,
      vix: 13.5,
      tape: 'risk_on',
      er: 0.72,
    },
    momentum: {
      session_id: new Date().toISOString().split('T')[0],
      setup_type: 'momentum_continuation',
      risk_off: false,
      mc_size_mult: 0.75,
      rs_vs_spy: 2.2,
      is_day_trade: true,
      flatten_hour: 15,
      entry_pattern: 'continuation',
    },
    would_entry_price: 245.80,
    would_stop_loss: 238.00,
    would_take_profit: 265.00,
    would_size_shares: 80,
    would_size_mult: 0.75,
    extra: {
      thesis: 'Strong momentum signal with excellent news catalyst (delivery numbers beat). However, vetoed due to high correlation with existing NVDA position — both ride AI/tech sentiment.',
      risks: [
        'VETOED: Portfolio correlation guard triggered — adding TSLA would concentrate tech exposure',
        'Already exposed to AI/semiconductor theme via NVDA',
        'RSI at 71 approaching overbought territory'
      ],
      sources: [
        { name: 'Tesla IR', tier: 1, headline: 'Q3 deliveries exceed expectations' },
        { name: 'Bloomberg', tier: 1, headline: 'Tesla China sales surge 40% YoY' },
        { name: 'Reuters', tier: 1, headline: 'EV maker posts record quarterly production' }
      ]
    },
  },
]

// ============================================================================
// Other Demo Data
// ============================================================================

export const DEMO_STATUS: BotStatus = {
  status: 'success',
  bot_running: false,
  mode: 'not_started',
  positions_count: 0,
  schwab_connected: false,
  uptime: '0:00:00',
}

export const DEMO_SYSTEM_STATS: SystemStats = {
  status: 'success',
  profile: {
    name: 'demo',
    confirmation_timeout_sec: 30,
    confirmation_timeout_action: 'cancel',
    require_close_confirmation: true,
    flatten_on_circuit: true,
  },
  news_bus: null,
  news_loop: null,
  supervisor: null,
  last_wake_reason: null,
}

export const DEMO_MARKET_INDICES: MarketIndicesResponse = {
  indices: [
    { symbol: 'SPY', name: 'S&P 500', last: 523.45, change_pct: 0.42 },
    { symbol: 'DIA', name: 'Dow Jones', last: 398.12, change_pct: 0.28 },
    { symbol: 'QQQ', name: 'Nasdaq', last: 448.90, change_pct: 0.65 },
    { symbol: 'IWM', name: 'Russell 2000', last: 208.34, change_pct: -0.18 },
    { symbol: '$VIX', name: 'VIX', last: 14.85, change_pct: -2.35 },
  ],
  last_updated: new Date().toISOString(),
  stale: true,
  error: null,
}

export const DEMO_ACCOUNT_STATS: AccountStats = {
  status: 'success',
  source: 'none',
  stats: {
    balance: 0,
    buying_power: 0,
    daily_pnl: 0,
    total_pnl: 0,
    position_count: 0,
    cash: 0,
  },
}

export const DEMO_POSITIONS: PositionsResponse = {
  status: 'success',
  count: 2,
  positions: [
    {
      symbol: 'NVDA',
      side: 'long',
      strategy: 'momentum_breakout',
      mode: 'simulation',
      entry_time: new Date(Date.now() - 3600000).toISOString(),
      entry_price: 885.20,
      current_price: 892.45,
      quantity: 50,
      stop_loss: 870.00,
      take_profit: 920.00,
      trailing_stop: null,
      scaled_out: false,
      unrealized_pnl: 362.50,
      updated_at: new Date().toISOString(),
      managed_by_bot: true,
      is_stale: true,
    },
    {
      symbol: 'META',
      side: 'long',
      strategy: 'trend_following',
      mode: 'simulation',
      entry_time: new Date(Date.now() - 7200000).toISOString(),
      entry_price: 502.80,
      current_price: 508.15,
      quantity: 30,
      stop_loss: 490.00,
      take_profit: 530.00,
      trailing_stop: 498.00,
      scaled_out: true,
      unrealized_pnl: 160.50,
      updated_at: new Date().toISOString(),
      managed_by_bot: true,
      is_stale: true,
    },
  ],
}

export const DEMO_DECISION_SNAPSHOTS: DecisionSnapshotsResponse = {
  status: 'success',
  count: DEMO_DECISION_CARDS.length,
  snapshots: DEMO_DECISION_CARDS,
}

export const DEMO_NEWS_BUS: NewsBusResponse = {
  status: 'success',
  symbol: 'NVDA',
  as_of: new Date().toISOString(),
  item_count: 4,
  aggregate: {
    article_count: 4,
    avg_sentiment: 0.65,
    high_impact_count: 1,
  },
  items: [
    {
      id: 'news-001',
      symbol: 'NVDA',
      headline: 'NVIDIA announces expanded cloud partnership',
      summary: 'Major cloud providers increase AI infrastructure spending',
      source: 'Reuters',
      source_tier: 1,
      sentiment: 0.8,
      is_high_impact: true,
      published_at: new Date(Date.now() - 1200000).toISOString(),
      fetched_at: new Date(Date.now() - 1000000).toISOString(),
      age_sec: 1200,
    },
    {
      id: 'news-002',
      symbol: 'NVDA',
      headline: 'Chip stocks rally on AI infrastructure spending',
      summary: null,
      source: 'Bloomberg',
      source_tier: 1,
      sentiment: 0.7,
      is_high_impact: false,
      published_at: new Date(Date.now() - 2400000).toISOString(),
      fetched_at: new Date(Date.now() - 2200000).toISOString(),
      age_sec: 2400,
    },
  ],
  has_high_impact: true,
}

export const DEMO_WS_UPDATE: WsDashboardUpdate = {
  type: 'dashboard_update',
  data: {
    account: {
      balance: 0,
      buying_power: 0,
      daily_pnl: 0,
      margin_call: false,
      cash: 0,
    },
    simulated_positions: DEMO_POSITIONS.positions.map(p => ({
      symbol: p.symbol,
      quantity: p.quantity,
      entry_price: p.entry_price,
      current_price: p.current_price,
      stop_loss: p.stop_loss,
      take_profit: p.take_profit,
      unrealized_pnl: p.unrealized_pnl,
      is_stale: true,
      type: 'simulated' as const,
      mode: 'simulation',
      managed_by_bot: p.managed_by_bot,
    })),
    real_positions: [],
    trades: [],
    screener: [
      { symbol: 'NVDA', last: 892.45, change: 2.15, volume: 45000000, volatility: 0.028, high: 895.00, low: 880.50 },
      { symbol: 'TSLA', last: 245.80, change: 3.42, volume: 82000000, volatility: 0.042, high: 248.00, low: 238.20 },
      { symbol: 'AAPL', last: 178.32, change: -0.45, volume: 52000000, volatility: 0.018, high: 180.00, low: 177.50 },
      { symbol: 'META', last: 508.15, change: 1.28, volume: 18000000, volatility: 0.025, high: 510.00, low: 502.00 },
      { symbol: 'MSFT', last: 415.60, change: 0.82, volume: 22000000, volatility: 0.015, high: 417.00, low: 412.50 },
    ],
    live_quotes: {},
    symbol_intel: [],
  },
}

// Helper to check if using demo mode
export function isDemoMode(): boolean {
  return true // This is used when we know we're in demo mode
}

// Get a random decision card for stream simulation
export function getRandomDecisionCard(): DecisionSnapshot {
  const idx = Math.floor(Math.random() * DEMO_DECISION_CARDS.length)
  return {
    ...DEMO_DECISION_CARDS[idx],
    ts: new Date().toISOString(),
    snapshot_id: `demo-${Date.now()}`,
  }
}
