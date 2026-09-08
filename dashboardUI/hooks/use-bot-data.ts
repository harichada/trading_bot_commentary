"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
  fetchAccountStats,
  fetchBotStatus,
  fetchMarketIndices,
  fetchPositions,
  botWsUrl,
  type AccountStats,
  type BotStatus,
  type MarketIndices,
  type PositionsResponse,
  type Commentary,
  type ScreenerItem,
  type LiveQuote,
  type WsMessage,
} from "@/lib/api"

export interface Poll<T> {
  data: T | null
  error: string | null
  /** true until the first successful (or failed) fetch resolves */
  loading: boolean
  /** true once at least one fetch has succeeded — drives "is this real data?" */
  live: boolean
}

/** Poll a fetcher on an interval; keeps last-good data across transient errors. */
function usePolling<T>(fetcher: () => Promise<T>, intervalMs: number): Poll<T> {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [live, setLive] = useState(false)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>

    const tick = async () => {
      try {
        const result = await fetcherRef.current()
        if (cancelled) return
        setData(result)
        setError(null)
        setLive(true)
      } catch (e) {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        // keep last-good data; do not clear it on a transient failure
      } finally {
        if (!cancelled) {
          setLoading(false)
          timer = setTimeout(tick, intervalMs)
        }
      }
    }

    tick()
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [intervalMs])

  return { data, error, loading, live }
}

export const useAccountStats = (intervalMs = 5000): Poll<AccountStats> =>
  usePolling(fetchAccountStats, intervalMs)

export const useBotStatus = (intervalMs = 5000): Poll<BotStatus> =>
  usePolling(fetchBotStatus, intervalMs)

export const useMarketIndices = (intervalMs = 10000): Poll<MarketIndices> =>
  usePolling(fetchMarketIndices, intervalMs)

export const usePositions = (intervalMs = 5000): Poll<PositionsResponse> =>
  usePolling(fetchPositions, intervalMs)

/**
 * Subscribe to the bot's /ws stream with auto-reconnect (exponential backoff).
 * onMessage receives each parsed JSON payload. Returns connection state.
 */
export function useBotSocket(onMessage: (msg: unknown) => void): { connected: boolean } {
  const [connected, setConnected] = useState(false)
  const onMessageRef = useRef(onMessage)
  onMessageRef.current = onMessage

  useEffect(() => {
    let ws: WebSocket | null = null
    let closedByUs = false
    let attempt = 0
    let reconnectTimer: ReturnType<typeof setTimeout>

    const connect = () => {
      try {
        ws = new WebSocket(botWsUrl())
      } catch {
        scheduleReconnect()
        return
      }
      ws.onopen = () => {
        attempt = 0
        setConnected(true)
      }
      ws.onmessage = (ev) => {
        try {
          onMessageRef.current(JSON.parse(ev.data))
        } catch {
          /* ignore non-JSON frames */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closedByUs) scheduleReconnect()
      }
      ws.onerror = () => ws?.close()
    }

    const scheduleReconnect = () => {
      attempt += 1
      const delay = Math.min(1000 * 2 ** attempt, 15000)
      reconnectTimer = setTimeout(connect, delay)
    }

    connect()
    return () => {
      closedByUs = true
      clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  return { connected }
}

/** Format a number as USD with sign, mono-friendly. */
export const fmtUsd = (n: number | null | undefined, signed = false): string => {
  if (n == null || Number.isNaN(n)) return "—"
  const sign = signed ? (n >= 0 ? "+" : "−") : n < 0 ? "−" : ""
  return `${sign}$${Math.abs(n).toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`
}

/** Format a percent change with sign. */
export const fmtPct = (n: number | null | undefined, signed = true): string => {
  if (n == null || Number.isNaN(n)) return "—"
  const sign = signed ? (n >= 0 ? "+" : "") : ""
  return `${sign}${n.toFixed(2)}%`
}

// ── WebSocket hooks for live data ──────────────────────────────────────────

const MAX_COMMENTARY_HISTORY = 50

export interface WsCommentaryState {
  items: Commentary[]
  connected: boolean
}

/**
 * Subscribe to live commentary messages from the bot's WebSocket.
 * Returns the commentary history (capped at MAX_COMMENTARY_HISTORY) and connection state.
 */
export function useWsCommentary(): WsCommentaryState {
  const [items, setItems] = useState<Commentary[]>([])
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let ws: WebSocket | null = null
    let closedByUs = false
    let attempt = 0
    let reconnectTimer: ReturnType<typeof setTimeout>

    const connect = () => {
      try {
        ws = new WebSocket(botWsUrl())
      } catch {
        scheduleReconnect()
        return
      }
      ws.onopen = () => {
        attempt = 0
        setConnected(true)
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as WsMessage
          if (msg.type === "commentary" && "data" in msg) {
            const commentary = msg.data as Commentary
            setItems((prev) => {
              const next = [commentary, ...prev]
              return next.slice(0, MAX_COMMENTARY_HISTORY)
            })
          }
        } catch {
          /* ignore non-JSON or malformed */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closedByUs) scheduleReconnect()
      }
      ws.onerror = () => ws?.close()
    }

    const scheduleReconnect = () => {
      attempt += 1
      const delay = Math.min(1000 * 2 ** attempt, 15000)
      reconnectTimer = setTimeout(connect, delay)
    }

    connect()
    return () => {
      closedByUs = true
      clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  return { items, connected }
}

export interface WsDashboardState {
  screener: ScreenerItem[]
  liveQuotes: Record<string, LiveQuote>
  connected: boolean
}

/**
 * Subscribe to dashboard_update messages from the bot's WebSocket.
 * Returns screener data and live quotes for the ticker bar.
 */
export function useWsDashboard(): WsDashboardState {
  const [screener, setScreener] = useState<ScreenerItem[]>([])
  const [liveQuotes, setLiveQuotes] = useState<Record<string, LiveQuote>>({})
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    let ws: WebSocket | null = null
    let closedByUs = false
    let attempt = 0
    let reconnectTimer: ReturnType<typeof setTimeout>

    const connect = () => {
      try {
        ws = new WebSocket(botWsUrl())
      } catch {
        scheduleReconnect()
        return
      }
      ws.onopen = () => {
        attempt = 0
        setConnected(true)
      }
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg.type === "dashboard_update" && msg.data) {
            if (Array.isArray(msg.data.screener)) {
              setScreener(msg.data.screener)
            }
            if (msg.data.live_quotes && typeof msg.data.live_quotes === "object") {
              setLiveQuotes(msg.data.live_quotes)
            }
          }
        } catch {
          /* ignore */
        }
      }
      ws.onclose = () => {
        setConnected(false)
        if (!closedByUs) scheduleReconnect()
      }
      ws.onerror = () => ws?.close()
    }

    const scheduleReconnect = () => {
      attempt += 1
      const delay = Math.min(1000 * 2 ** attempt, 15000)
      reconnectTimer = setTimeout(connect, delay)
    }

    connect()
    return () => {
      closedByUs = true
      clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [])

  return { screener, liveQuotes, connected }
}
