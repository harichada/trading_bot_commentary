import { useEffect, useRef, useCallback, useState } from 'react'

type WsHandler = (msg: Record<string, unknown>) => void

export function useWebSocket(handlers: Record<string, WsHandler>) {
  const [connected, setConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const handlersRef = useRef(handlers)
  handlersRef.current = handlers

  useEffect(() => {
    let alive = true
    let ws: WebSocket

    const connect = () => {
      if (!alive) return
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws

      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (alive) setTimeout(connect, 2000)
      }
      ws.onerror = () => { /* swallow */ }
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          const type = msg.type as string
          if (type && handlersRef.current[type]) {
            handlersRef.current[type](msg)
          }
          // Always fire '*' catch-all if present
          if (handlersRef.current['*']) {
            handlersRef.current['*'](msg)
          }
        } catch { /* ignore */ }
      }
    }

    connect()

    // Keepalive ping every 15s
    const ping = setInterval(() => {
      if (ws?.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 15000)

    return () => {
      alive = false
      clearInterval(ping)
      ws?.close()
    }
  }, [])

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  return { connected, send }
}
