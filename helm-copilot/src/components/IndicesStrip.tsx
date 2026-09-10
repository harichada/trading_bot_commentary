import { clsx } from 'clsx'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import type { MarketIndex } from '../api/types'

interface IndicesStripProps {
  indices: MarketIndex[]
  isStale: boolean
  className?: string
}

function formatPrice(price: number): string {
  if (price >= 1000) {
    return price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
  }
  return price.toFixed(2)
}

function formatChange(pct: number): string {
  const sign = pct >= 0 ? '+' : ''
  return `${sign}${pct.toFixed(2)}%`
}

interface IndexChipProps {
  index: MarketIndex
  isStale: boolean
}

function IndexChip({ index, isStale }: IndexChipProps) {
  const isPositive = index.change_pct >= 0
  const isFlat = Math.abs(index.change_pct) < 0.05
  const isVix = index.symbol === '$VIX'
  
  return (
    <div
      className={clsx(
        'flex items-center gap-2 px-3 py-1.5 rounded-md',
        'bg-helm-surface/50 border border-helm-border/50',
        isStale && 'opacity-60'
      )}
    >
      <span className="text-xs text-zinc-400 font-medium">
        {index.name}
      </span>
      
      <span className={clsx(
        'mono-nums text-sm font-medium',
        isStale && 'text-zinc-500'
      )}>
        {formatPrice(index.last)}
      </span>
      
      <div className={clsx(
        'flex items-center gap-0.5 text-xs font-medium',
        isFlat && 'text-zinc-500',
        !isFlat && isPositive && !isVix && 'text-emerald-400',
        !isFlat && !isPositive && !isVix && 'text-red-400',
        // VIX is inverted: high VIX = bad, low VIX = good
        !isFlat && isPositive && isVix && 'text-red-400',
        !isFlat && !isPositive && isVix && 'text-emerald-400',
      )}>
        {isFlat ? (
          <Minus className="w-3 h-3" />
        ) : isPositive ? (
          <TrendingUp className="w-3 h-3" />
        ) : (
          <TrendingDown className="w-3 h-3" />
        )}
        <span className="mono-nums">{formatChange(index.change_pct)}</span>
      </div>
    </div>
  )
}

export function IndicesStrip({ indices, isStale, className }: IndicesStripProps) {
  if (indices.length === 0) {
    return (
      <div className={clsx('flex items-center gap-2 px-4 py-2 overflow-x-auto', className)}>
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="h-8 w-32 rounded-md skeleton"
          />
        ))}
      </div>
    )
  }

  return (
    <div className={clsx(
      'flex items-center gap-2 px-4 py-2 overflow-x-auto',
      'bg-helm-bg border-b border-helm-border',
      className
    )}>
      {indices.map((index) => (
        <IndexChip key={index.symbol} index={index} isStale={isStale} />
      ))}
      
      {isStale && (
        <span className="text-xs text-amber-400 ml-2 whitespace-nowrap">
          Data may be delayed
        </span>
      )}
    </div>
  )
}
