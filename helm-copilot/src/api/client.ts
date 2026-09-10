/**
 * Helm API Client
 * 
 * Defensive HTTP client for the trading bot API.
 * Falls back to demo fixtures when API is unreachable.
 */

import type {
  BotStatus,
  SystemStats,
  MarketIndicesResponse,
  AccountStats,
  PositionsResponse,
  DecisionsResponse,
  DecisionSnapshotsResponse,
  NewsBusResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:9000'
const API_KEY = import.meta.env.VITE_API_KEY

interface FetchOptions extends RequestInit {
  timeout?: number
}

class ApiError extends Error {
  constructor(
    message: string,
    public status?: number,
    public isOffline: boolean = false
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function fetchWithTimeout(
  url: string,
  options: FetchOptions = {}
): Promise<Response> {
  const { timeout = 8000, ...fetchOptions } = options
  
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeout)
  
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(API_KEY ? { Authorization: `Bearer ${API_KEY}` } : {}),
    ...options.headers,
  }
  
  try {
    const response = await fetch(url, {
      ...fetchOptions,
      headers,
      signal: controller.signal,
    })
    clearTimeout(id)
    return response
  } catch (error) {
    clearTimeout(id)
    if (error instanceof Error && error.name === 'AbortError') {
      throw new ApiError('Request timeout', undefined, true)
    }
    throw new ApiError('Network error', undefined, true)
  }
}

async function apiGet<T>(endpoint: string, timeout?: number): Promise<T> {
  const url = `${API_BASE}${endpoint}`
  const response = await fetchWithTimeout(url, { timeout })
  
  if (!response.ok) {
    throw new ApiError(
      `HTTP ${response.status}: ${response.statusText}`,
      response.status
    )
  }
  
  return response.json()
}

// Track connectivity state
let _isOnline = true
let _lastCheckTime = 0
const CHECK_INTERVAL = 5000

export async function checkConnectivity(): Promise<boolean> {
  const now = Date.now()
  if (now - _lastCheckTime < CHECK_INTERVAL) {
    return _isOnline
  }
  _lastCheckTime = now
  
  try {
    await fetchWithTimeout(`${API_BASE}/api/status`, { timeout: 3000 })
    _isOnline = true
    return true
  } catch {
    _isOnline = false
    return false
  }
}

export function isOnline(): boolean {
  return _isOnline
}

// ============================================================================
// API Endpoints
// ============================================================================

export async function getStatus(): Promise<BotStatus> {
  return apiGet<BotStatus>('/api/status')
}

export async function getSystemStats(): Promise<SystemStats> {
  return apiGet<SystemStats>('/api/system-stats')
}

export async function getMarketIndices(): Promise<MarketIndicesResponse> {
  return apiGet<MarketIndicesResponse>('/api/market-indices')
}

export async function getAccountStats(): Promise<AccountStats> {
  return apiGet<AccountStats>('/api/account-stats')
}

export async function getPositions(): Promise<PositionsResponse> {
  return apiGet<PositionsResponse>('/api/positions/db')
}

export async function getDecisions(params?: {
  date?: string
  symbol?: string
  component?: string
  limit?: number
}): Promise<DecisionsResponse> {
  const searchParams = new URLSearchParams()
  if (params?.date) searchParams.set('date', params.date)
  if (params?.symbol) searchParams.set('symbol', params.symbol)
  if (params?.component) searchParams.set('component', params.component)
  if (params?.limit) searchParams.set('limit', params.limit.toString())
  
  const query = searchParams.toString()
  return apiGet<DecisionsResponse>(`/api/decisions${query ? `?${query}` : ''}`)
}

export async function getDecisionSnapshots(params?: {
  symbol?: string
  strategy_id?: string
  action?: string
  limit?: number
  since?: string
}): Promise<DecisionSnapshotsResponse> {
  const searchParams = new URLSearchParams()
  if (params?.symbol) searchParams.set('symbol', params.symbol)
  if (params?.strategy_id) searchParams.set('strategy_id', params.strategy_id)
  if (params?.action) searchParams.set('action', params.action)
  if (params?.limit) searchParams.set('limit', params.limit.toString())
  if (params?.since) searchParams.set('since', params.since)
  
  const query = searchParams.toString()
  return apiGet<DecisionSnapshotsResponse>(`/api/decision-snapshots${query ? `?${query}` : ''}`)
}

export async function getNewsBus(
  symbol: string,
  maxAgeSec: number = 14400
): Promise<NewsBusResponse> {
  return apiGet<NewsBusResponse>(
    `/api/news-bus/${symbol.toUpperCase()}?max_age_sec=${maxAgeSec}`
  )
}

// Probe for optional routes — returns null if not available
export async function probeRoute<T>(endpoint: string): Promise<T | null> {
  try {
    return await apiGet<T>(endpoint, 3000)
  } catch {
    return null
  }
}

// ============================================================================
// WebSocket Connection
// ============================================================================

export interface WsConnectionOptions {
  onMessage: (data: unknown) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Error) => void
  token?: string
}

export function createWebSocket(options: WsConnectionOptions): {
  connect: () => void
  disconnect: () => void
  isConnected: () => boolean
} {
  let ws: WebSocket | null = null
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  let reconnectAttempts = 0
  const maxReconnectAttempts = 10
  const baseReconnectDelay = 1000
  
  const getWsUrl = () => {
    const base = API_BASE.replace(/^http/, 'ws')
    const url = new URL('/ws', base)
    if (options.token) {
      url.searchParams.set('token', options.token)
    }
    return url.toString()
  }
  
  const connect = () => {
    if (ws?.readyState === WebSocket.OPEN) return
    
    try {
      ws = new WebSocket(getWsUrl())
      
      ws.onopen = () => {
        reconnectAttempts = 0
        _isOnline = true
        options.onConnect?.()
      }
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          options.onMessage(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }
      
      ws.onclose = () => {
        options.onDisconnect?.()
        scheduleReconnect()
      }
      
      ws.onerror = (event) => {
        options.onError?.(new Error('WebSocket error'))
        console.error('WebSocket error:', event)
      }
    } catch (error) {
      options.onError?.(error instanceof Error ? error : new Error('Failed to connect'))
      scheduleReconnect()
    }
  }
  
  const scheduleReconnect = () => {
    if (reconnectAttempts >= maxReconnectAttempts) {
      _isOnline = false
      return
    }
    
    const delay = Math.min(
      baseReconnectDelay * Math.pow(2, reconnectAttempts),
      30000
    )
    
    reconnectTimeout = setTimeout(() => {
      reconnectAttempts++
      connect()
    }, delay)
  }
  
  const disconnect = () => {
    if (reconnectTimeout) {
      clearTimeout(reconnectTimeout)
      reconnectTimeout = null
    }
    if (ws) {
      ws.close()
      ws = null
    }
  }
  
  const isConnected = () => ws?.readyState === WebSocket.OPEN
  
  return { connect, disconnect, isConnected }
}

export { ApiError }
