import { useEffect, useRef, useState, useCallback } from 'react'
import { WSManager } from '../api/websocket'

interface UseWebSocketResult {
  lastMessage: Record<string, unknown> | null
  connected: boolean
  messages: Record<string, unknown>[]
}

function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}

export function useWebSocket(path: string, maxMessages = 100): UseWebSocketResult {
  const [lastMessage, setLastMessage] = useState<Record<string, unknown> | null>(null)
  const [connected, setConnected] = useState(false)
  const [messages, setMessages] = useState<Record<string, unknown>[]>([])
  const wsRef = useRef<WSManager | null>(null)

  const handleMessage = useCallback((data: Record<string, unknown>) => {
    setLastMessage(data)
    setMessages(prev => {
      const next = [...prev, data]
      return next.length > maxMessages ? next.slice(-maxMessages) : next
    })
  }, [maxMessages])

  const handleStatus = useCallback((status: boolean) => {
    setConnected(status)
  }, [])

  useEffect(() => {
    const url = wsUrl(path)
    const ws = new WSManager({
      url,
      onMessage: handleMessage,
      onStatusChange: handleStatus,
    })
    wsRef.current = ws
    ws.connect()

    return () => {
      ws.disconnect()
      wsRef.current = null
    }
  }, [path, handleMessage, handleStatus])

  return { lastMessage, connected, messages }
}
