/** Rudra Trading Engine — API client */

const BASE = '/api'

// API key from env or localStorage
function getApiKey(): string {
  return localStorage.getItem('rudra_api_key') ?? ''
}

export function setApiKey(key: string) {
  localStorage.setItem('rudra_api_key', key)
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options?.headers as Record<string, string>),
  }
  const apiKey = getApiKey()
  if (apiKey) headers['X-API-Key'] = apiKey

  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text.slice(0, 200)}`)
  }
  const data = await res.json()
  // Backend sometimes returns { error: ... } with 200 status
  if (data && typeof data === 'object' && 'error' in data && Object.keys(data).length === 1) {
    throw new Error(data.error)
  }
  return data
}

export interface HealthData {
  status: string
  version: string
  uptime_seconds: number
  equity: number
  daily_pnl: number
  open_positions: number
  trading_halted: boolean
  trader_status: string
}

export interface TraderState {
  status: string
  equity: number
  daily_pnl: number
  daily_pnl_pct: number
  total_pnl: number
  win_rate: number
  profit_factor: number
  max_drawdown: number
  trades_today: number
  total_trades: number
  positions: Position[]
  candidates: Candidate[]
  messages: Message[]
}

export interface Position {
  symbol: string
  direction: string
  shares: number
  remaining_shares: number
  entry_price: number
  current_price: number
  stop_price: number
  pnl: number
  pnl_pct: number
  entry_time: string
  strategy_id: string
}

export interface Candidate {
  symbol: string
  gap_pct: number
  score: number
  direction: string
  premarket_price: number
  prev_close: number
  vol_ratio: number
  catalyst: string
}

export interface Message {
  time: string
  type: string
  text: string
}

export interface Trade {
  trade_id: number
  symbol: string
  entry_price: number
  exit_price: number
  shares: number
  pnl: number
  pnl_pct: number
  entry_time: string
  exit_time: string
  exit_reason: string
  holding_minutes: number
  side: string
  strategy_id: string
}

export interface EquityCurve {
  dates: string[]
  equity: number[]
  drawdown: number[]
}

export const api = {
  health: () => request<HealthData>('/health'),
  state: () => request<Record<string, unknown>>('/state'),
  trades: () => request<{ trades: Trade[] }>('/trades').then(r => r.trades),
  equityCurve: () => request<EquityCurve>('/equity_curve').catch(() => ({ dates: [], equity: [], drawdown: [] })),
  strategies: () => request<{ strategies: Array<{ id: string; name: string; description: string }>; active: string }>('/strategies').catch(() => ({ strategies: [], active: '' })),
  scan: () => request<unknown>('/scan', { method: 'POST' }),
  pause: () => request<unknown>('/pause', { method: 'POST' }),
  resume: () => request<unknown>('/resume', { method: 'POST' }),
  stop: () => request<unknown>('/stop', { method: 'POST' }),
  reset: () => request<unknown>('/reset', { method: 'POST' }),
  start: () => request<unknown>('/start', { method: 'POST' }),
}
