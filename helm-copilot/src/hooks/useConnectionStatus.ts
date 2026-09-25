import { useState, useEffect, useCallback } from 'react'
import { checkConnectivity, isOnline } from '../api/client'
import type { ConnectionState, ConnectionInfo } from '../api/types'

const CHECK_INTERVAL = 10000 // 10 seconds

export function useConnectionStatus(): ConnectionInfo & {
  refresh: () => Promise<void>
} {
  const [state, setState] = useState<ConnectionState>('connecting')
  const [lastConnected, setLastConnected] = useState<Date | null>(null)
  const [lastError, setLastError] = useState<string | null>(null)
  const [isDemo, setIsDemo] = useState(false)

  const refresh = useCallback(async () => {
    try {
      const online = await checkConnectivity()
      if (online) {
        setState('connected')
        setLastConnected(new Date())
        setLastError(null)
        setIsDemo(false)
      } else {
        setState('offline')
        setIsDemo(true)
      }
    } catch (error) {
      setState('offline')
      setLastError(error instanceof Error ? error.message : 'Connection failed')
      setIsDemo(true)
    }
  }, [])

  useEffect(() => {
    refresh()
    const interval = setInterval(refresh, CHECK_INTERVAL)
    return () => clearInterval(interval)
  }, [refresh])

  // Also check on visibility change
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        refresh()
      }
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange)
  }, [refresh])

  return {
    state,
    lastConnected,
    lastError,
    isDemo,
    refresh,
  }
}

export function useIsOnline(): boolean {
  const [online, setOnline] = useState(isOnline())

  useEffect(() => {
    const interval = setInterval(() => {
      setOnline(isOnline())
    }, 2000)
    return () => clearInterval(interval)
  }, [])

  return online
}
