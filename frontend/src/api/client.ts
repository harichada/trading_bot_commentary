const GF_BASE = '/gf-api'
const ID_BASE = '/id-api'

export interface ApiError {
  status: number
  message: string
  isAuth: boolean
  isOffline: boolean
}

interface FetchOptions extends RequestInit {
  timeout?: number
}

async function request<T>(url: string, opts: FetchOptions = {}): Promise<T> {
  const { timeout = 10000, ...fetchOpts } = opts
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)

  let response: Response
  try {
    response = await fetch(url, {
      ...fetchOpts,
      signal: controller.signal,
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        ...fetchOpts.headers,
      },
    })
  } catch (err) {
    clearTimeout(id)
    const error: ApiError = {
      status: 0,
      message: 'Service unreachable',
      isAuth: false,
      isOffline: true,
    }
    throw error
  }
  clearTimeout(id)

  if (response.status === 401 || response.status === 403) {
    const error: ApiError = {
      status: response.status,
      message: 'Authentication required',
      isAuth: true,
      isOffline: false,
    }
    throw error
  }

  if (!response.ok) {
    const error: ApiError = {
      status: response.status,
      message: `HTTP ${response.status}: ${response.statusText}`,
      isAuth: false,
      isOffline: false,
    }
    throw error
  }
  return response.json() as Promise<T>
}

export function isApiError(err: unknown): err is ApiError {
  return typeof err === 'object' && err !== null && 'isOffline' in err && 'isAuth' in err
}

// ---------- Gap Fade API (port 8003) ----------

export interface GfHealth {
  status: string
  trader_status?: string
  positions?: number
  uptime_seconds?: number
  trading_halted?: boolean
  equity?: number
  daily_pnl?: number
  [key: string]: unknown
}

export interface GfPosition {
  symbol: string
  direction: string
  entry_price: number
  current_price?: number
  qty: number
  unrealized_pnl?: number
  pnl_pct?: number
  stop_price?: number
  target_price?: number
  entry_time?: string
  [key: string]: unknown
}

export interface GfTrade {
  symbol: string
  direction: string
  entry_price: number
  exit_price: number
  qty: number
  pnl: number
  entry_time: string
  exit_time: string
  exit_reason?: string
  [key: string]: unknown
}

export interface GfState {
  positions?: GfPosition[]
  trades?: GfTrade[]
  equity?: number
  daily_pnl?: number
  starting_equity?: number
  win_count?: number
  loss_count?: number
  [key: string]: unknown
}

export interface GfConfigSchema {
  [key: string]: {
    type: string
    default?: unknown
    min?: number
    max?: number
    description?: string
    [key: string]: unknown
  }
}

export const gfApi = {
  getHealth: () => request<GfHealth>(`${GF_BASE}/health`),
  getState: () => request<GfState>(`${GF_BASE}/state`),
  getCombinedState: () => request<GfState>(`${GF_BASE}/state/combined`),
  getConfig: () => request<Record<string, unknown>>(`${GF_BASE}/config`),
  getConfigSchema: () => request<GfConfigSchema>(`${GF_BASE}/config/schema`),
  getConfigHistory: () => request<Record<string, unknown>[]>(`${GF_BASE}/config/history`),
  updateConfig: (config: Record<string, unknown>) =>
    request<Record<string, unknown>>(`${GF_BASE}/config`, {
      method: 'POST',
      body: JSON.stringify(config),
    }),
  getBrokers: () => request<Record<string, unknown>[]>(`${GF_BASE}/brokers`),
  getBacktestResults: () =>
    request<Record<string, unknown>>(`${GF_BASE}/backtest/results`),
  getBacktestStatus: () =>
    request<Record<string, unknown>>(`${GF_BASE}/backtest/status`),
  runBacktest: (params: Record<string, unknown>) =>
    request<Record<string, unknown>>(`${GF_BASE}/backtest`, {
      method: 'POST',
      body: JSON.stringify(params),
    }),
  startTrading: () =>
    request<Record<string, unknown>>(`${GF_BASE}/start`, { method: 'POST' }),
  stopTrading: () =>
    request<Record<string, unknown>>(`${GF_BASE}/stop`, { method: 'POST' }),
}

// ---------- Intraday API (port 8004) ----------

export interface IdStrategy {
  strategy_id: string
  name?: string
  enabled: boolean
  status?: string
  trades_today?: number
  pnl_today?: number
  win_rate?: number
  [key: string]: unknown
}

export interface IdPosition {
  symbol: string
  strategy_id?: string
  direction?: string
  qty: number
  entry_price: number
  current_price?: number
  unrealized_pnl?: number
  pnl_pct?: number
  [key: string]: unknown
}

export interface IdTrade {
  symbol: string
  strategy_id?: string
  side: string
  price: number
  qty: number
  time: string
  pnl?: number
  [key: string]: unknown
}

export interface IdHealth {
  status: string
  trading_active?: boolean
  strategies_enabled?: number
  positions?: number
  [key: string]: unknown
}

export const idApi = {
  getHealth: () => request<IdHealth>(`${ID_BASE}/health`),
  getPositions: () => request<IdPosition[]>(`${ID_BASE}/positions`),
  getTrades: () => request<IdTrade[]>(`${ID_BASE}/trades`),
  getStrategies: () => request<IdStrategy[]>(`${ID_BASE}/strategies`),
  enableStrategy: (id: string) =>
    request<Record<string, unknown>>(`${ID_BASE}/strategies/${id}/enable`, { method: 'POST' }),
  disableStrategy: (id: string) =>
    request<Record<string, unknown>>(`${ID_BASE}/strategies/${id}/disable`, { method: 'POST' }),
  submitOrder: (order: Record<string, unknown>) =>
    request<Record<string, unknown>>(`${ID_BASE}/order`, {
      method: 'POST',
      body: JSON.stringify(order),
    }),
  start: () => request<Record<string, unknown>>(`${ID_BASE}/start`, { method: 'POST' }),
  stop: () => request<Record<string, unknown>>(`${ID_BASE}/stop`, { method: 'POST' }),
}
