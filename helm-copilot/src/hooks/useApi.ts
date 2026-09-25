import { useState, useEffect, useCallback, useRef } from 'react'
import * as api from '../api/client'
import * as demo from '../fixtures/demo'
import type {
  BotStatus,
  SystemStats,
  MarketIndicesResponse,
  AccountStats,
  PositionsResponse,
  DecisionSnapshotsResponse,
  NewsBusResponse,
} from '../api/types'

interface UseApiState<T> {
  data: T | null
  isLoading: boolean
  error: string | null
  isDemo: boolean
  asOf: Date | null
}

interface UseApiOptions {
  pollInterval?: number
  enabled?: boolean
}

function useApiCall<T>(
  fetcher: () => Promise<T>,
  fallback: T,
  options: UseApiOptions = {}
): UseApiState<T> & { refetch: () => Promise<void> } {
  const { pollInterval, enabled = true } = options
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    isLoading: true,
    error: null,
    isDemo: false,
    asOf: null,
  })
  const mountedRef = useRef(true)

  const fetch = useCallback(async () => {
    if (!enabled) return
    
    try {
      const data = await fetcher()
      if (mountedRef.current) {
        setState({
          data,
          isLoading: false,
          error: null,
          isDemo: false,
          asOf: new Date(),
        })
      }
    } catch (error) {
      if (mountedRef.current) {
        setState({
          data: fallback,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          isDemo: true,
          asOf: new Date(),
        })
      }
    }
  }, [fetcher, fallback, enabled])

  useEffect(() => {
    mountedRef.current = true
    fetch()
    
    let interval: ReturnType<typeof setInterval> | undefined
    if (pollInterval && pollInterval > 0) {
      interval = setInterval(fetch, pollInterval)
    }
    
    return () => {
      mountedRef.current = false
      if (interval) clearInterval(interval)
    }
  }, [fetch, pollInterval])

  return { ...state, refetch: fetch }
}

// ============================================================================
// Specific API Hooks
// ============================================================================

export function useStatus(options?: UseApiOptions) {
  return useApiCall<BotStatus>(
    api.getStatus,
    demo.DEMO_STATUS,
    options
  )
}

export function useSystemStats(options?: UseApiOptions) {
  return useApiCall<SystemStats>(
    api.getSystemStats,
    demo.DEMO_SYSTEM_STATS,
    options
  )
}

export function useMarketIndices(options?: UseApiOptions) {
  return useApiCall<MarketIndicesResponse>(
    api.getMarketIndices,
    demo.DEMO_MARKET_INDICES,
    { pollInterval: 10000, ...options }
  )
}

export function useAccountStats(options?: UseApiOptions) {
  return useApiCall<AccountStats>(
    api.getAccountStats,
    demo.DEMO_ACCOUNT_STATS,
    options
  )
}

export function usePositions(options?: UseApiOptions) {
  return useApiCall<PositionsResponse>(
    api.getPositions,
    demo.DEMO_POSITIONS,
    { pollInterval: 5000, ...options }
  )
}

export function useDecisionSnapshots(
  params?: {
    symbol?: string
    strategy_id?: string
    action?: string
    limit?: number
    since?: string
  },
  options?: UseApiOptions
) {
  const fetcher = useCallback(
    () => api.getDecisionSnapshots(params),
    [params?.symbol, params?.strategy_id, params?.action, params?.limit, params?.since]
  )
  
  return useApiCall<DecisionSnapshotsResponse>(
    fetcher,
    demo.DEMO_DECISION_SNAPSHOTS,
    { pollInterval: 5000, ...options }
  )
}

export function useNewsBus(
  symbol: string,
  maxAgeSec: number = 14400,
  options?: UseApiOptions
) {
  const fetcher = useCallback(
    () => api.getNewsBus(symbol, maxAgeSec),
    [symbol, maxAgeSec]
  )
  
  return useApiCall<NewsBusResponse>(
    fetcher,
    { ...demo.DEMO_NEWS_BUS, symbol },
    options
  )
}
