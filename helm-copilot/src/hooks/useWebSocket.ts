import { useState, useEffect, useCallback, useRef } from 'react'
import { createWebSocket } from '../api/client'
import { DEMO_WS_UPDATE, getRandomDecisionCard } from '../fixtures/demo'
import type { WsDashboardUpdate, WsMessage, DecisionSnapshot } from '../api/types'

interface UseWebSocketOptions {
  token?: string
  enabled?: boolean
  onDecisionCard?: (card: DecisionSnapshot) => void
}

interface UseWebSocketState {
  connected: boolean
  isDemo: boolean
  lastUpdate: WsDashboardUpdate | null
  lastUpdateTime: Date | null
}

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketState {
  const { token, enabled = true, onDecisionCard } = options
  const [state, setState] = useState<UseWebSocketState>({
    connected: false,
    isDemo: false,
    lastUpdate: null,
    lastUpdateTime: null,
  })
  
  const wsRef = useRef<ReturnType<typeof createWebSocket> | null>(null)
  const demoIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const handleMessage = useCallback((data: unknown) => {
    const message = data as WsMessage
    
    if (message.type === 'dashboard_update') {
      setState(prev => ({
        ...prev,
        lastUpdate: message as WsDashboardUpdate,
        lastUpdateTime: new Date(),
      }))
    }
  }, [])

  const startDemoMode = useCallback(() => {
    setState(prev => ({
      ...prev,
      connected: false,
      isDemo: true,
      lastUpdate: DEMO_WS_UPDATE,
      lastUpdateTime: new Date(),
    }))

    // Simulate occasional decision card updates in demo mode
    if (onDecisionCard) {
      demoIntervalRef.current = setInterval(() => {
        // Random chance to emit a decision card (roughly once per 30 seconds)
        if (Math.random() < 0.03) {
          onDecisionCard(getRandomDecisionCard())
        }
      }, 1000)
    }
  }, [onDecisionCard])

  const stopDemoMode = useCallback(() => {
    if (demoIntervalRef.current) {
      clearInterval(demoIntervalRef.current)
      demoIntervalRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!enabled) return

    const ws = createWebSocket({
      token,
      onMessage: handleMessage,
      onConnect: () => {
        stopDemoMode()
        setState(prev => ({
          ...prev,
          connected: true,
          isDemo: false,
        }))
      },
      onDisconnect: () => {
        setState(prev => ({
          ...prev,
          connected: false,
        }))
        // Start demo mode after a brief delay if still disconnected
        setTimeout(() => {
          if (!wsRef.current?.isConnected()) {
            startDemoMode()
          }
        }, 5000)
      },
      onError: () => {
        startDemoMode()
      },
    })

    wsRef.current = ws
    ws.connect()

    // If not connected after 5 seconds, switch to demo mode
    const demoTimeout = setTimeout(() => {
      if (!ws.isConnected()) {
        startDemoMode()
      }
    }, 5000)

    return () => {
      clearTimeout(demoTimeout)
      stopDemoMode()
      ws.disconnect()
      wsRef.current = null
    }
  }, [enabled, token, handleMessage, startDemoMode, stopDemoMode])

  return state
}

// Hook for just tracking connection state
export function useWsConnectionState(options: UseWebSocketOptions = {}): {
  connected: boolean
  isDemo: boolean
} {
  const { connected, isDemo } = useWebSocket(options)
  return { connected, isDemo }
}
