import { clsx } from 'clsx'
import { Clock, Activity, Zap, AlertTriangle } from 'lucide-react'
import { LiveBadge, StaleBadge, DemoBadge } from './ui/Badge'

interface TrustStripProps {
  isLive: boolean
  isDemo: boolean
  isStale: boolean
  asOf: Date | null
  botRunning: boolean
  mode: string
  className?: string
}

function formatAsOf(date: Date | null): string {
  if (!date) return '—'
  const now = new Date()
  const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000)
  
  if (diffSec < 5) return 'just now'
  if (diffSec < 60) return `${diffSec}s ago`
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`
  return date.toLocaleTimeString()
}

export function TrustStrip({
  isLive,
  isDemo,
  isStale,
  asOf,
  botRunning,
  mode,
  className,
}: TrustStripProps) {
  return (
    <div
      className={clsx(
        'flex items-center justify-between gap-4 px-4 py-2',
        'bg-helm-bg/50 border-t border-helm-border',
        'text-xs text-zinc-400',
        className
      )}
    >
      {/* Left: Connection status */}
      <div className="flex items-center gap-3">
        {isDemo ? (
          <DemoBadge />
        ) : isLive && !isStale ? (
          <LiveBadge />
        ) : isStale ? (
          <StaleBadge />
        ) : (
          <span className="text-zinc-500">Connecting...</span>
        )}
        
        <div className="flex items-center gap-1.5 text-zinc-500">
          <Clock className="w-3 h-3" />
          <span>as-of: {formatAsOf(asOf)}</span>
        </div>
      </div>

      {/* Center: Mode indicator */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5">
          <Activity className="w-3 h-3 text-zinc-500" />
          <span className={clsx(
            botRunning ? 'text-emerald-400' : 'text-zinc-500'
          )}>
            {botRunning ? 'Bot Active' : 'Bot Stopped'}
          </span>
        </div>
        
        <div className="flex items-center gap-1.5">
          <Zap className="w-3 h-3 text-zinc-500" />
          <span className={clsx(
            mode === 'live' && 'text-cyan-400',
            mode === 'simulation' && 'text-amber-400',
            mode === 'not_started' && 'text-zinc-500'
          )}>
            {mode === 'live' ? 'LIVE' : mode === 'simulation' ? 'SIM' : 'OFF'}
          </span>
        </div>
      </div>

      {/* Right: Disclaimer */}
      <div className="flex items-center gap-1.5 text-zinc-600">
        <AlertTriangle className="w-3 h-3" />
        <span className="hidden sm:inline">
          Informational/educational only — not financial advice
        </span>
        <span className="sm:hidden">Not advice</span>
      </div>
    </div>
  )
}
