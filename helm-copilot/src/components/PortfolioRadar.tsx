import { clsx } from 'clsx'
import {
  Compass,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Shield,
  Target,
  Clock,
  AlertCircle,
  Bot,
  Hand,
} from 'lucide-react'
import { usePositions, useAccountStats, useWebSocket } from '../hooks'
import { Card, CardContent, CardHeader } from './ui/Card'
import { Badge, DemoBadge, LiveBadge, StaleBadge, AsOfChip } from './ui/Badge'
import type { Position } from '../api/types'

function safeNumber(val: unknown, fallback = 0): number {
  if (typeof val === 'number' && isFinite(val)) return val
  return fallback
}

function formatCurrency(value: number): string {
  const safe = safeNumber(value)
  return safe.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  })
}

function formatPercent(value: number): string {
  const safe = safeNumber(value)
  const sign = safe >= 0 ? '+' : ''
  return `${sign}${safe.toFixed(2)}%`
}

function formatTime(ts: string | null): string {
  if (!ts) return '—'
  try {
    const date = new Date(ts)
    if (isNaN(date.getTime())) return '—'
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    })
  } catch {
    return '—'
  }
}

interface PositionRowProps {
  position: Position
}

function PositionRow({ position }: PositionRowProps) {
  const entryPrice = safeNumber(position.entry_price)
  const currentPrice = safeNumber(position.current_price)
  const unrealizedPnl = safeNumber(position.unrealized_pnl)
  
  const pnlPercent = entryPrice > 0
    ? ((currentPrice - entryPrice) / entryPrice) * 100
    : 0
  const isPositive = unrealizedPnl >= 0
  const isLong = position.side === 'long'
  const isManaged = position.managed_by_bot !== false
  const isLive = position.mode === 'live'

  return (
    <div className={clsx(
      'p-3 border-b border-helm-border/50 last:border-0',
      'hover:bg-helm-elevated/20 transition-colors',
      position.is_stale && 'opacity-60',
      // Visual affordance for managed vs hands-off
      isManaged ? 'position-managed' : 'position-unmanaged',
    )}>
      <div className="flex items-center justify-between gap-3">
        {/* Left: Symbol and Details */}
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={clsx(
            'p-1.5 rounded flex-shrink-0',
            isPositive ? 'bg-emerald-500/15' : 'bg-red-500/15'
          )}>
            {isPositive ? (
              <TrendingUp className="w-4 h-4 text-emerald-400" />
            ) : (
              <TrendingDown className="w-4 h-4 text-red-400" />
            )}
          </div>
          
          <div className="min-w-0">
            <div className="flex items-center gap-1.5 flex-wrap">
              <span className="text-base font-semibold text-white tracking-tight">
                {position.symbol || '—'}
              </span>
              <Badge variant={isLong ? 'success' : 'error'} size="sm">
                {isLong ? 'L' : 'S'}
              </Badge>
              <Badge variant={isLive ? 'live' : 'default'} size="sm">
                {isLive ? 'LIVE' : 'SIM'}
              </Badge>
              {!isManaged && (
                <span className="text-[10px] text-zinc-500 flex items-center gap-0.5" title="Not managed by bot">
                  <Hand className="w-3 h-3" />
                </span>
              )}
              {position.is_stale && <StaleBadge compact />}
            </div>
            
            <div className="flex items-center gap-2 mt-0.5 text-[11px] text-zinc-500">
              <span className="mono-nums">{position.quantity} × ${entryPrice.toFixed(2)}</span>
              {position.strategy && (
                <span className="text-zinc-600 truncate max-w-[100px]">{position.strategy}</span>
              )}
            </div>
          </div>
        </div>
        
        {/* Right: P&L */}
        <div className="text-right flex-shrink-0">
          <span className="mono-nums text-base font-medium text-white leading-data block">
            ${currentPrice.toFixed(2)}
          </span>
          <div className="flex items-center justify-end gap-1.5">
            <span className={clsx(
              'mono-nums text-sm font-semibold leading-data',
              isPositive ? 'text-emerald-400' : 'text-red-400'
            )}>
              {isPositive ? '+' : ''}{formatCurrency(unrealizedPnl)}
            </span>
            <span className={clsx(
              'mono-nums text-[10px]',
              isPositive ? 'text-emerald-500/60' : 'text-red-500/60'
            )}>
              {formatPercent(pnlPercent)}
            </span>
          </div>
        </div>
      </div>
      
      {/* Stop/Target row (collapsed, only if present) */}
      {(position.stop_loss > 0 || position.take_profit > 0) && (
        <div className="flex items-center gap-3 mt-1.5 ml-9 text-[10px]">
          {position.stop_loss > 0 && (
            <span className="flex items-center gap-0.5 text-red-400/80">
              <Shield className="w-2.5 h-2.5" />
              ${safeNumber(position.stop_loss).toFixed(2)}
            </span>
          )}
          {position.take_profit > 0 && (
            <span className="flex items-center gap-0.5 text-emerald-400/80">
              <Target className="w-2.5 h-2.5" />
              ${safeNumber(position.take_profit).toFixed(2)}
            </span>
          )}
          {position.entry_time && (
            <span className="flex items-center gap-0.5 text-zinc-600">
              <Clock className="w-2.5 h-2.5" />
              {formatTime(position.entry_time)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

export function PortfolioRadar() {
  const { data: positionsData, isLoading, isDemo, asOf, refetch } = usePositions()
  const { data: accountData } = useAccountStats({ pollInterval: 10000 })
  const { lastUpdate } = useWebSocket()

  // Prefer WebSocket data for real-time positions if available
  const wsPositions = lastUpdate?.data?.simulated_positions ?? []
  const wsRealPositions = lastUpdate?.data?.real_positions ?? []
  const positions = positionsData?.positions ?? []

  // Combine positions, preferring REST positions over WS fallback
  const allPositions: Position[] = positions.length > 0 ? positions : [
    ...wsPositions.map(p => ({
      symbol: p.symbol,
      side: 'long' as const,
      strategy: null,
      mode: p.mode || 'simulation',
      entry_time: null,
      entry_price: safeNumber(p.entry_price),
      current_price: safeNumber(p.current_price),
      quantity: safeNumber(p.quantity),
      stop_loss: safeNumber(p.stop_loss),
      take_profit: safeNumber(p.take_profit),
      trailing_stop: null,
      scaled_out: false,
      unrealized_pnl: safeNumber(p.unrealized_pnl),
      updated_at: new Date().toISOString(),
      managed_by_bot: p.managed_by_bot,
      is_stale: p.is_stale,
    })),
    ...wsRealPositions.map(p => ({
      symbol: p.symbol,
      side: 'long' as const,
      strategy: null,
      mode: 'live' as const,
      entry_time: null,
      entry_price: safeNumber(p.entry_price),
      current_price: safeNumber(p.current_price),
      quantity: safeNumber(p.quantity),
      stop_loss: safeNumber(p.stop_loss),
      take_profit: safeNumber(p.take_profit),
      trailing_stop: null,
      scaled_out: false,
      unrealized_pnl: safeNumber(p.unrealized_pnl),
      updated_at: new Date().toISOString(),
      managed_by_bot: p.managed_by_bot,
      is_stale: p.is_stale,
      is_long_term: p.is_long_term,
      day_pnl: safeNumber(p.day_pnl),
      pnl_percent: safeNumber(p.pnl_percent),
      market_value: safeNumber(p.market_value),
    })),
  ]

  // Split positions by mode for cleaner display
  const livePositions = allPositions.filter(p => p.mode === 'live')
  const simPositions = allPositions.filter(p => p.mode !== 'live')

  const totalPnl = allPositions.reduce((sum, p) => sum + safeNumber(p.unrealized_pnl), 0)
  const isPositive = totalPnl >= 0

  const stats = accountData?.stats ?? {
    balance: 0,
    buying_power: 0,
    daily_pnl: 0,
    total_pnl: totalPnl,
    position_count: allPositions.length,
    cash: 0,
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Compass className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-white">Radar</h2>
          {isDemo && <DemoBadge />}
        </div>
        
        <div className="flex items-center gap-2">
          <AsOfChip time={asOf} />
          <button
            onClick={() => refetch()}
            className="p-1.5 rounded text-zinc-500 hover:text-white hover:bg-helm-elevated transition-colors"
            aria-label="Refresh"
          >
            <RefreshCw className={clsx('w-3.5 h-3.5', isLoading && 'animate-spin')} />
          </button>
        </div>
      </div>

      {/* Account Summary — tighter */}
      <Card className="mb-3">
        <CardContent className="py-3">
          <div className="grid grid-cols-4 gap-3">
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wide">Value</span>
              <p className="mono-nums text-lg font-bold text-white leading-data">
                {formatCurrency(safeNumber(stats.balance))}
              </p>
            </div>
            
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wide">Day P&L</span>
              <p className={clsx(
                'mono-nums text-lg font-bold leading-data',
                safeNumber(stats.daily_pnl) >= 0 ? 'text-emerald-400' : 'text-red-400'
              )}>
                {safeNumber(stats.daily_pnl) >= 0 ? '+' : ''}{formatCurrency(safeNumber(stats.daily_pnl))}
              </p>
            </div>
            
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wide">Unreal</span>
              <p className={clsx(
                'mono-nums text-lg font-bold leading-data',
                isPositive ? 'text-emerald-400' : 'text-red-400'
              )}>
                {isPositive ? '+' : ''}{formatCurrency(totalPnl)}
              </p>
            </div>
            
            <div>
              <span className="text-[10px] text-zinc-500 uppercase tracking-wide">BP</span>
              <p className="mono-nums text-lg font-bold text-white leading-data">
                {formatCurrency(safeNumber(stats.buying_power))}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Live Positions (if any) */}
      {livePositions.length > 0 && (
        <Card className="mb-3">
          <CardHeader className="py-2 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-medium text-white">Live</h3>
              <Badge variant="live" size="sm">{livePositions.length}</Badge>
            </div>
            <LiveBadge compact />
          </CardHeader>
          <div>
            {livePositions.map((position, i) => (
              <PositionRow key={`live-${position.symbol}-${i}`} position={position} />
            ))}
          </div>
        </Card>
      )}

      {/* Simulated Positions */}
      <Card>
        <CardHeader className="py-2 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-medium text-white">
              {livePositions.length > 0 ? 'Simulated' : 'Positions'}
            </h3>
            <Badge variant="default" size="sm">{simPositions.length || allPositions.length}</Badge>
          </div>
          {!isDemo && simPositions.length > 0 && (
            <span className="flex items-center gap-1 text-[10px] text-zinc-500">
              <Bot className="w-3 h-3" />
              paper
            </span>
          )}
        </CardHeader>
        
        {isLoading && allPositions.length === 0 ? (
          <CardContent className="py-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-16 mb-2 rounded skeleton" />
            ))}
          </CardContent>
        ) : (livePositions.length === 0 && simPositions.length === 0) ? (
          <div className="empty-state">
            <Compass className="empty-state-icon" />
            <p className="empty-state-title">No positions</p>
            <p className="empty-state-subtitle">
              {isDemo ? 'Demo data will appear when fixtures load' : 'Positions appear when the bot opens trades'}
            </p>
          </div>
        ) : (
          <div>
            {(livePositions.length > 0 ? simPositions : allPositions).map((position, i) => (
              <PositionRow key={`pos-${position.symbol}-${i}`} position={position} />
            ))}
          </div>
        )}
      </Card>

      {/* Data Source Notice — more subtle */}
      {accountData?.source && accountData.source !== 'none' && (
        <div className="mt-3 flex items-center gap-1.5 text-[10px] text-zinc-600">
          <AlertCircle className="w-3 h-3" />
          <span>
            {accountData.source === 'schwab' ? 'Schwab live' : accountData.source}
          </span>
        </div>
      )}

      {/* Demo Notice — more subtle */}
      {isDemo && allPositions.length > 0 && (
        <div className="mt-4 px-3 py-2.5 rounded bg-purple-500/8 border border-purple-500/15">
          <p className="text-xs text-purple-400/90">
            <strong className="font-semibold">Demo:</strong> Simulated positions. Connect to API for live data.
          </p>
        </div>
      )}
    </div>
  )
}
