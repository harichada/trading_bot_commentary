import { useCallback } from 'react'

export function useFormatters() {
  const formatCurrency = useCallback((value: number): string => {
    const abs = Math.abs(value)
    const prefix = value < 0 ? '-$' : '$'
    return `${prefix}${abs.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
  }, [])

  const formatPercent = useCallback((value: number): string => {
    const sign = value > 0 ? '+' : ''
    return `${sign}${value.toFixed(2)}%`
  }, [])

  const formatNumber = useCallback((value: number, decimals = 2): string => {
    return value.toLocaleString('en-US', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })
  }, [])

  const formatCompact = useCallback((value: number): string => {
    if (Math.abs(value) >= 1_000_000_000) {
      return `$${(value / 1_000_000_000).toFixed(2)}B`
    }
    if (Math.abs(value) >= 1_000_000) {
      return `$${(value / 1_000_000).toFixed(2)}M`
    }
    if (Math.abs(value) >= 1_000) {
      return `$${(value / 1_000).toFixed(1)}K`
    }
    return `$${value.toFixed(2)}`
  }, [])

  return { formatCurrency, formatPercent, formatNumber, formatCompact }
}
