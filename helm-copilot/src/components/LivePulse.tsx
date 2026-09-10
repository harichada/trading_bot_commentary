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
  Zap,
  ChevronRight,
  RefreshCw
} from 'lucide-react'
import { useDecisionSnapshots, useWebSocket } from '../hooks'
import { Card, CardContent } from './ui/Card'
import { Badge, DemoBadge } from './ui/Badge'
import type { DecisionSnapshot } from '../api/types'

interface DecisionCardPreviewProps {
  decision: DecisionSnapshot
  onClick: () => void
}

function getVerdictInfo(action: string): { 
  label: string
  variant: 'act' | 'wait' | 'nothing'
  Icon: typeof TrendingUp
} {
  switch (action) {
    case 'signal_buy':
      return { label: 'BUY', variant: 'act', Icon: TrendingUp }
    case 'signal_sell':
      return { label: 'SELL', variant: 'act', Icon: TrendingDown }
    case 'skip':
      return { label: 'WAIT', variant: 'wait', Icon: Pause }
    case 'veto':
      return { label: 'DO NOTHING', variant: 'nothing', Icon: XCircle }
    default:
      return { label: action.toUpperCase(), variant: 'nothing', Icon: Pause }
  }
}

function formatTimeAgo(ts: string): string {
  const now = new Date()
  const then = new Date(ts)
  const diffSec = Math.floor((now.getTime() - then.getTime()) / 1000)
  
  if (diffSec < 60) return `${diffSec}s ago`
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`
  return then.toLocaleDateString()
}

function DecisionCardPreview({ decision, onClick }: DecisionCardPreviewProps) {
  const { label, variant, Icon } = getVerdictInfo(decision.action)
  const isDemo = decision.mode === 'demo'
  
  return (
    <Card interactive onClick={onClick} className="animate-slide-in">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          {/* Left: Symbol and Verdict */}
          <div className="flex items-start gap-3">
            <div className={clsx(
              'p-2 rounded-lg',
              variant === 'act' && 'bg-cyan-500/20',
              variant === 'wait' && 'bg-amber-500/20',
              variant === 'nothing' && 'bg-zinc-500/20',
            )}>
              <Icon className={clsx(
                'w-5 h-5',
                variant === 'act' && 'text-cyan-400',
                variant === 'wait' && 'text-amber-400',
                variant === 'nothing' && 'text-zinc-400',
              )} />
            </div>
            
            <div>
              <div className="flex items-center gap-2">
                <span className="text-lg font-semibold text-white">
                  {decision.symbol}
                </span>
                <Badge variant={variant}>{label}</Badge>
                {isDemo && <DemoBadge />}
              </div>
              
              <p className="text-sm text-zinc-400 mt-0.5">
                {decision.reason.replace(/_/g, ' ')}
              </p>
              
              <div className="flex items-center gap-3 mt-2 text-xs text-zinc-500">
                <span className="flex items-center gap-1">
                  <Zap className="w-3 h-3" />
                  {decision.strategy_id}
                </span>
                <span className="flex items-center gap-1">
                  <Clock className="w-3 h-3" />
                  {formatTimeAgo(decision.ts)}
                </span>
              </div>
            </div>
          </div>
          
          {/* Right: Key Numbers */}
          <div className="flex flex-col items-end gap-1">
            <span className="mono-nums text-lg font-medium text-white">
              ${decision.price_vol.price.toFixed(2)}
            </span>
            
            <div className="flex items-center gap-2 text-xs">
              <span className={clsx(
                'mono-nums',
                decision.confidence >= 0.7 && 'text-emerald-400',
                decision.confidence >= 0.5 && decision.confidence < 0.7 && 'text-amber-400',
                decision.confidence < 0.5 && 'text-zinc-500',
              )}>
                {(decision.confidence * 100).toFixed(0)}% conf
              </span>
            </div>
            
            <ChevronRight className="w-4 h-4 text-zinc-500 mt-2" />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export function LivePulse() {
  const navigate = useNavigate()
  const [decisions, setDecisions] = useState<DecisionSnapshot[]>([])
  const { data, isLoading, isDemo, refetch } = useDecisionSnapshots(
    { limit: 20 },
    { pollInterval: 5000 }
  )

  // Handle incoming decision cards from WebSocket
  const handleNewDecision = useCallback((card: DecisionSnapshot) => {
    setDecisions(prev => [card, ...prev.slice(0, 49)])
  }, [])

  useWebSocket({
    onDecisionCard: handleNewDecision,
  })

  // Combine API data with WebSocket data
  const allDecisions = [...decisions, ...(data?.snapshots ?? [])]
    .filter((d, i, arr) => arr.findIndex(x => x.snapshot_id === d.snapshot_id) === i)
    .slice(0, 50)

  const handleCardClick = (decision: DecisionSnapshot) => {
    navigate(`/decision/${decision.snapshot_id}`, { state: { decision } })
  }

  return (
    <div className="p-4 max-w-3xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Radio className="w-5 h-5 text-cyan-400" />
          <h2 className="text-lg font-semibold text-white">Live Pulse</h2>
          {isDemo && <DemoBadge />}
        </div>
        
        <button
          onClick={() => refetch()}
          className="p-2 rounded-md text-zinc-400 hover:text-white hover:bg-helm-elevated transition-colors"
        >
          <RefreshCw className={clsx('w-4 h-4', isLoading && 'animate-spin')} />
        </button>
      </div>

      {/* Decision Stream */}
      <div className="space-y-3">
        {allDecisions.length === 0 && isLoading ? (
          // Loading skeletons
          Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-24 rounded-lg skeleton" />
          ))
        ) : allDecisions.length === 0 ? (
          <Card>
            <CardContent className="py-12 text-center">
              <Radio className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
              <p className="text-zinc-400">No recent decisions</p>
              <p className="text-xs text-zinc-500 mt-1">
                Decisions will appear here as the bot analyzes opportunities
              </p>
            </CardContent>
          </Card>
        ) : (
          allDecisions.map((decision) => (
            <DecisionCardPreview
              key={decision.snapshot_id}
              decision={decision}
              onClick={() => handleCardClick(decision)}
            />
          ))
        )}
      </div>

      {/* Demo Mode Notice */}
      {isDemo && (
        <div className="mt-6 p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
          <p className="text-sm text-purple-400">
            <strong>Demo Mode:</strong> These are simulated decision cards demonstrating the UI. 
            Connect to a running trading bot to see live decisions.
          </p>
        </div>
      )}
    </div>
  )
}
