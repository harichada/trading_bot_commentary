import { useState, useEffect, useRef, useCallback } from 'react'
import { api, type TraderState, type HealthData } from '../lib/api'

/** Map raw API response to our TraderState shape */
function mapState(raw: Record<string, unknown>): TraderState {
  const metrics = (raw.metrics ?? {}) as Record<string, unknown>
  const daily = (raw.daily_stats ?? {}) as Record<string, unknown>
  const posObj = (raw.positions ?? {}) as Record<string, unknown>

  // positions can be dict (keyed by symbol) or array
  const positions = Array.isArray(posObj)
    ? posObj
    : Object.values(posObj).map((p: unknown) => p as TraderState['positions'][0])

  const candidates = Array.isArray(raw.candidates) ? raw.candidates : []
  const messages = Array.isArray(raw.messages) ? raw.messages : []

  // max_drawdown_pct is already in percent (e.g. 34.62 means 34.62%)
  // max_drawdown is the dollar amount
  const equity = Number(raw.equity ?? 0)
  const peakEquity = Number(raw.peak_equity ?? equity)
  const rawDdPct = Number(metrics.max_drawdown_pct ?? 0)
  const maxDdPct = rawDdPct > 0
    ? rawDdPct / 100  // convert 34.62 → 0.3462 for display as "34.6%"
    : peakEquity > 0 ? Number(metrics.max_drawdown ?? 0) / peakEquity : 0

  // daily_pnl_pct may be null — compute from equity if needed
  const dailyPnl = Number(daily.pnl ?? 0)
  const dailyPnlPct = daily.pnl_pct != null
    ? Number(daily.pnl_pct)
    : equity > 0 ? dailyPnl / equity : 0

  return {
    status: String(raw.status ?? 'stopped'),
    equity,
    daily_pnl: dailyPnl,
    daily_pnl_pct: dailyPnlPct,
    total_pnl: Number(metrics.total_pnl ?? 0),
    win_rate: Number(metrics.win_rate ?? 0),
    profit_factor: Number(metrics.profit_factor ?? 0),
    max_drawdown: maxDdPct,
    trades_today: Number(daily.trades ?? 0),
    total_trades: Number(metrics.total_trades ?? 0),
    positions,
    candidates,
    messages,
  }
}

export function useTraderState(pollMs = 3000) {
  const [state, setState] = useState<TraderState | null>(null)
  const [health, setHealth] = useState<HealthData | null>(null)
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // WebSocket for real-time updates
  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => setConnected(true)
    ws.onclose = () => {
      setConnected(false)
      setTimeout(() => { wsRef.current = null }, 3000)
    }
    ws.onerror = () => { /* swallow */ }
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'state_update' && msg.state) {
          setState(mapState(msg.state))
        }
      } catch { /* ignore */ }
    }

    return () => { ws.close() }
  }, [])

  // Poll as fallback
  const fetchState = useCallback(async () => {
    try {
      const [rawState, h] = await Promise.allSettled([api.state(), api.health()])
      if (rawState.status === 'fulfilled') {
        setState(mapState(rawState.value as Record<string, unknown>))
      }
      if (h.status === 'fulfilled') {
        setHealth(h.value)
      }
      setConnected(rawState.status === 'fulfilled' || h.status === 'fulfilled')
    } catch {
      setConnected(false)
    }
  }, [])

  useEffect(() => {
    fetchState()
    const timer = setInterval(fetchState, pollMs)
    return () => clearInterval(timer)
  }, [fetchState, pollMs])

  return { state, health, connected, refresh: fetchState }
}
