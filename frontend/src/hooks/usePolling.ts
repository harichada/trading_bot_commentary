import { useState, useEffect, useCallback, useRef } from 'react'
import { type ApiError, isApiError } from '../api/client'

interface UsePollingResult<T> {
  data: T | null
  loading: boolean
  error: ApiError | null
  refresh: () => void
}

export function usePolling<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 10000,
): UsePollingResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ApiError | null>(null)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const doFetch = useCallback(async () => {
    try {
      const result = await fetcherRef.current()
      setData(result)
      setError(null)
    } catch (err) {
      if (isApiError(err)) {
        setError(err)
      } else {
        setError({
          status: 0,
          message: String(err),
          isAuth: false,
          isOffline: true,
        })
      }
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    doFetch()
    const timer = setInterval(doFetch, intervalMs)
    return () => clearInterval(timer)
  }, [doFetch, intervalMs])

  return { data, loading, error, refresh: doFetch }
}
