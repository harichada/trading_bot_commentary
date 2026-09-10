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
} from 'lucide-react'
import { usePositions, useAccountStats, useWebSocket } from '../hooks'
import { Card, CardContent, CardHeader } from './ui/Card'
import { Badge, DemoBadge, LiveBadge, StaleBadge } from './ui/Badge'
import type { Position } from '../api/types'

function formatCurrency(value: number): string {
  return value.toLocaleString('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
  })
}

function formatPercent(value: number): string {
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(2)}%`
}

function formatTime(ts: string | null): string {
  if (!ts) return '—'
  const date = new Date(ts)
  return date.toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    hour12: true,
  })
}

interface PositionRowProps {
  position: Position
}

function PositionRow({ position }: PositionRowProps) {
  const pnlPercent = position.entry_price > 0
    ? ((position.current_price - position.entry_price) / position.entry_price) * 100
    : 0
  const isPositive = position.unrealized_pnl >= 0
  const isLong = position.side === 'long'

  return (
    <div className={clsx(
      'p-4 border-b border-helm-border last:border-0',
      'hover:bg-helm-elevated/30 transition-colors',
      position.is_stale && 'opacity-70'
    )}>
      <div className="flex items-start justify-between gap-4">
        {/* Left: Symbol and Details */}
        <div className="flex items-start gap-3">
          <div className={clsx(
            'p-2 rounded-lg',
            isPositive ? 'bg-emerald-500/20' : 'bg-red-500/20'
          )}>
            {isPositive ? (
              <TrendingUp className="w-5 h-5 text-emerald-400" />
            ) : (
              <TrendingDown className="w-5 h-5 text-red-400" />
            )}
          </div>
          
          <div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold text-white">
                {position.symbol}
              </span>
              <Badge variant={isLong ? 'success' : 'error'} size="sm">
                {isLong ? 'LONG' : 'SHORT'}
              </Badge>
              <Badge variant={position.mode === 'live' ? 'live' : 'stale'} size="sm">
                {position.mode === 'live' ? 'LIVE' : 'SIM'}
              </Badge>
              {position.is_stale && <StaleBadge />}
            </div>
            
            <div className="flex items-center gap-3 mt-1 text-xs text-zinc-500">
              <span>{position.quantity} shares</span>
              <span>@ ${position.entry_price.toFixed(2)}</span>
              {position.strategy && (
                <span className="text-zinc-600">• {position.strategy}</span>
              )}
            </div>
            
            <div className="flex items-center gap-3 mt-2 text-xs">
              {position.stop_loss > 0 && (
                <span className="flex items-center gap-1 text-red-400">
                  <Shield className="w-3 h-3" />
                  SL: ${position.stop_loss.toFixed(2)}
                </span>
              )}
              {position.take_profit > 0 && (
                <span className="flex items-center gap-1 text-emerald-400">
                  <Target className="w-3 h-3" />
                  TP: ${position.take_profit.toFixed(2)}
                </span>
              )}
              {position.entry_time && (
                <span className="flex items-center gap-1 text-zinc-500">
                  <Clock className="w-3 h-3" />
                  {formatTime(position.entry_time)}
                </span>
              )}
            </div>
          </div>
        </div>
        
        {/* Right: P&L */}
        <div className="text-right">
          <div className="flex flex-col items-end">
            <span className="mono-nums text-lg font-medium text-white">
              ${position.current_price.toFixed(2)}
            </span>
            <span className={clsx(
              'mono-nums text-sm font-medium',
              isPositive ? 'text-emerald-400' : 'text-red-400'
            )}>
              {isPositive ? '+' : ''}{formatCurrency(position.unrealized_pnl)}
            </span>
            <span className={clsx(
              'mono-nums text-xs',
              isPositive ? 'text-emerald-400/70' : 'text-red-400/70'
            )}>
              {formatPercent(pnlPercent)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export function PortfolioRadar() {
  const { data: positionsData, isLoading, isDemo, refetch } = usePositions()
  const { data: accountData } = useAccountStats({ pollInterval: 10000 })
  const { lastUpdate } = useWebSocket()

  // Prefer WebSocket data for real-time positions if available
  const wsPositions = lastUpdate?.data?.simulated_positions ?? []
  const wsRealPositions = lastUpdate?.data?.real_positions ?? []
  const positions = positionsData?.positions ?? []

  // Combine positions, preferring WebSocket data
  const allPositions: Position[] = positions.length > 0 ? positions : [
    ...wsPositions.map(p => ({
      symbol: p.symbol,
      side: 'long' as const,
      strategy: null,
      mode: p.mode || 'simulation',
      entry_time: null,
      entry_price: p.entry_price,
      current_price: p.current_price,
      quantity: p.quantity,
      stop_loss: p.stop_loss ?? 0,
      take_profit: p.take_profit ?? 0,
      trailing_stop: null,
      scaled_out: false,
      unrealized_pnl: p.unrealized_pnl,
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
      entry_price: p.entry_price,
      current_price: p.current_price,
      quantity: p.quantity,
      stop_loss: p.stop_loss ?? 0,
      take_profit: p.take_profit ?? 0,
      trailing_stop: null,
      scaled_out: false,
      unrealized_pnl: p.unrealized_pnl,
      updated_at: new Date().toISOString(),
      managed_by_bot: p.managed_by_bot,
      is_stale: p.is_stale,
      is_long_term: p.is_long_term,
      day_pnl: p.day_pnl,
      pnl_percent: p.pnl_percent,
      market_value: p.market_value,
    })),
  ]

  const totalPnl = allPositions.reduce((sum, p) => sum + (p.unrealized_pnl || 0), 0)
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
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Compass className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Portfolio Radar</h2>
          {isDemo && <DemoBadge />}
        </div>
        
        <button
          onClick={() => refetch()}
          className="p-2 rounded-md text-zinc-400 hover:text-white hover:bg-helm-elevated transition-colors"
        >
          <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
        </button>
      </div>

      {/* Account Summary */}
      <Card className="mb-4">
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <span className="text-xs text-zinc-500">Portfolio Value</span>
              <p className="mono-nums text-xl font-bold text-white">
                {formatCurrency(stats.balance)}
              </p>
            </div>
            
            <div>
              <span className="text-xs text-zinc-500">Day P&L</span>
              <p className={clsx(
                'mono-nums text-xl font-bold',
                stats.daily_pnl >= 0 ? 'text-emerald-400' : 'text-red-400'
              )}>
                {stats.daily_pnl >= 0 ? '+' : ''}{formatCurrency(stats.daily_pnl)}
              </p>
            </div>
            
            <div>
              <span className="text-xs text-zinc-500">Unrealized P&L</span>
              <p className={clsx(
                'mono-nums text-xl font-bold',
                isPositive ? 'text-emerald-400' : 'text-red-400'
              )}>
                {isPositive ? '+' : ''}{formatCurrency(totalPnl)}
              </p>
            </div>
            
            <div>
              <span className="text-xs text-zinc-500">Buying Power</span>
              <p className="mono-nums text-xl font-bold text-white">
                {formatCurrency(stats.buying_power)}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Positions List */}
      <Card>
        <CardHeader className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h3 className="font-medium text-white">Open Positions</h3>
            <Badge variant="default">
              {allPositions.length}
            </Badge>
          </div>
          
          {!isDemo && allPositions.length > 0 && <LiveBadge />}
        </CardHeader>
        
        {isLoading && allPositions.length === 0 ? (
          <CardContent>
            {Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="h-20 mb-3 rounded skeleton" />
            ))}
          </CardContent>
        ) : allPositions.length === 0 ? (
          <CardContent className="py-12 text-center">
            <Compass className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
            <p className="text-zinc-400">No open positions</p>
            <p className="text-xs text-zinc-500 mt-1">
              Positions will appear here when the bot opens trades
            </p>
          </CardContent>
        ) : (
          <div>
            {allPositions.map((position, i) => (
              <PositionRow
                key={`${position.symbol}-${i}`}
                position={position}
              />
            ))}
          </div>
        )}
      </Card>

      {/* Data Source Notice */}
      {accountData?.source && accountData.source !== 'none' && (
        <div className="mt-4 flex items-center gap-2 text-xs text-zinc-500">
          <AlertCircle className="w-3 h-3" />
          <span>
            Data source: {accountData.source === 'schwab' ? 'Schwab (live)' : accountData.source}
          </span>
        </div>
      )}

      {/* Demo Notice */}
      {isDemo && (
        <div className="mt-6 p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
          <p className="text-sm text-purple-400">
            <strong>Demo Mode:</strong> These are simulated positions demonstrating the UI. 
            Connect to a running trading bot to see live positions.
          </p>
        </div>
      )}
    </div>
  )
}
