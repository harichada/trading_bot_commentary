import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { clsx } from 'clsx'
import { 
  Radio, 
  TrendingUp, 
  TrendingDown, 
  Pause, 
  XCircle,
  Clock,
  ChevronRight,
  RefreshCw,
  Wifi,
  WifiOff,
} from 'lucide-react'
import { useDecisionSnapshots, useWebSocket } from '../hooks'
import { Card, CardContent } from './ui/Card'
import { Badge, DemoBadge, AsOfChip } from './ui/Badge'
import type { DecisionSnapshot } from '../api/types'

interface DecisionCardPreviewProps {
  decision: DecisionSnapshot
  onClick: () => void
  isNew?: boolean
}

function getVerdictInfo(action: string): { 
  label: string
  shortLabel: string
  variant: 'act' | 'wait' | 'nothing'
  Icon: typeof TrendingUp
} {
  switch (action) {
    case 'signal_buy':
      return { label: 'ACT', shortLabel: 'BUY', variant: 'act', Icon: TrendingUp }
    case 'signal_sell':
      return { label: 'ACT', shortLabel: 'SELL', variant: 'act', Icon: TrendingDown }
    case 'skip':
      return { label: 'WAIT', shortLabel: 'WAIT', variant: 'wait', Icon: Pause }
    case 'veto':
      return { label: 'PASS', shortLabel: 'PASS', variant: 'nothing', Icon: XCircle }
    default:
      return { label: action.toUpperCase(), shortLabel: action.toUpperCase(), variant: 'nothing', Icon: Pause }
  }
}

function formatTimeAgo(ts: string): string {
  try {
    const now = new Date()
    const then = new Date(ts)
    if (isNaN(then.getTime())) return '—'
    const diffSec = Math.floor((now.getTime() - then.getTime()) / 1000)
    
    if (diffSec < 5) return 'now'
    if (diffSec < 60) return `${diffSec}s`
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m`
    if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h`
    return then.toLocaleDateString()
  } catch {
    return '—'
  }
}

function safeNumber(val: unknown, fallback = 0): number {
  if (typeof val === 'number' && isFinite(val)) return val
  return fallback
}

function DecisionCardPreview({ decision, onClick, isNew }: DecisionCardPreviewProps) {
  const { label, shortLabel, variant, Icon } = getVerdictInfo(decision.action)
  const isDemo = decision.mode === 'demo'
  const price = safeNumber(decision.price_vol?.price)
  const confidence = safeNumber(decision.confidence)
  
  return (
    <Card 
      interactive 
      onClick={onClick} 
      className={clsx(isNew && 'stream-insert')}
    >
      <CardContent className="p-3 sm:p-4">
        <div className="flex items-center justify-between gap-3">
          {/* Left: Icon + Symbol + Verdict */}
          <div className="flex items-center gap-2.5 min-w-0">
            <div className={clsx(
              'p-2 rounded-md flex-shrink-0',
              variant === 'act' && 'bg-cyan-500/15',
              variant === 'wait' && 'bg-amber-500/10',
              variant === 'nothing' && 'bg-zinc-600/15',
            )}>
              <Icon className={clsx(
                'w-4 h-4',
                variant === 'act' && 'text-cyan-400',
                variant === 'wait' && 'text-amber-400',
                variant === 'nothing' && 'text-zinc-500',
              )} />
            </div>
            
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-base font-semibold text-white tracking-tight">
                  {decision.symbol || '—'}
                </span>
                <Badge variant={variant} size="sm">
                  {variant === 'act' ? shortLabel : label}
                </Badge>
                {isDemo && <DemoBadge compact />}
              </div>
              
              <div className="flex items-center gap-2 mt-0.5 text-[11px] text-zinc-500">
                <span className="truncate max-w-[140px]">
                  {(decision.reason || '').replace(/_/g, ' ')}
                </span>
                <span className="flex items-center gap-0.5 flex-shrink-0">
                  <Clock className="w-3 h-3" />
                  {formatTimeAgo(decision.ts)}
                </span>
              </div>
            </div>
          </div>
          
          {/* Right: Price + Confidence */}
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="text-right">
              <span className="mono-nums text-base font-medium text-white leading-data block">
                ${price.toFixed(2)}
              </span>
              <span className={clsx(
                'mono-nums text-[11px] leading-data',
                confidence >= 0.7 && 'text-emerald-400',
                confidence >= 0.5 && confidence < 0.7 && 'text-amber-400',
                confidence < 0.5 && 'text-zinc-500',
              )}>
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            
            <ChevronRight className="w-4 h-4 text-zinc-600" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function LivePulse() {
  const navigate = useNavigate()
  const [wsDecisions, setWsDecisions] = useState<DecisionSnapshot[]>([])
  const [newIds, setNewIds] = useState<Set<string>>(new Set())
  const { data, isLoading, isDemo, asOf, refetch } = useDecisionSnapshots(
    { limit: 20 },
    { pollInterval: 5000 }
  )

  // Handle incoming decision cards from WebSocket
  const handleNewDecision = useCallback((card: DecisionSnapshot) => {
    setWsDecisions(prev => [card, ...prev.slice(0, 49)])
    setNewIds(prev => new Set([...prev, card.snapshot_id]))
    // Clear "new" flag after animation
    setTimeout(() => {
      setNewIds(prev => {
        const next = new Set(prev)
        next.delete(card.snapshot_id)
        return next
      })
    }, 500)
  }, [])

  const { connected } = useWebSocket({
    onDecisionCard: handleNewDecision,
  })

  // Combine API data with WebSocket data, deduping
  const allDecisions = [...wsDecisions, ...(data?.snapshots ?? [])]
    .filter((d, i, arr) => arr.findIndex(x => x.snapshot_id === d.snapshot_id) === i)
    .slice(0, 50)

  const handleCardClick = (decision: DecisionSnapshot) => {
    navigate(`/decision/${decision.snapshot_id}`, { state: { decision } })
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Radio className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-white">Pulse</h2>
          {isDemo ? (
            <DemoBadge />
          ) : connected ? (
            <span className="flex items-center gap-1 text-[10px] text-emerald-500">
              <Wifi className="w-3 h-3" />
            </span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] text-zinc-500">
              <WifiOff className="w-3 h-3" />
            </span>
          )}
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

      {/* Decision Stream */}
      <div className="space-y-2">
        {allDecisions.length === 0 && isLoading ? (
          // Loading skeletons — more refined
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-[72px] rounded-lg skeleton" />
          ))
        ) : allDecisions.length === 0 ? (
          // Empty state — more polished
          <Card>
            <div className="empty-state">
              {isDemo ? (
                <>
                  <WifiOff className="empty-state-icon" />
                  <p className="empty-state-title">No API connection</p>
                  <p className="empty-state-subtitle">
                    Demo data will appear when fixtures load
                  </p>
                </>
              ) : (
                <>
                  <Radio className="empty-state-icon" />
                  <p className="empty-state-title">Waiting for signals</p>
                  <p className="empty-state-subtitle">
                    Decisions appear here as the bot evaluates opportunities
                  </p>
                </>
              )}
            </div>
          </Card>
        ) : (
          allDecisions.map((decision) => (
            <DecisionCardPreview
              key={decision.snapshot_id}
              decision={decision}
              onClick={() => handleCardClick(decision)}
              isNew={newIds.has(decision.snapshot_id)}
            />
          ))
        )}
      </div>

      {/* Demo Mode Notice — more subtle */}
      {isDemo && allDecisions.length > 0 && (
        <div className="mt-4 px-3 py-2.5 rounded bg-purple-500/8 border border-purple-500/15">
          <p className="text-xs text-purple-400/90">
            <strong className="font-semibold">Demo:</strong> Simulated decisions. Connect to API for live data.
          </p>
        </div>
      )}
    </div>
  )
}
