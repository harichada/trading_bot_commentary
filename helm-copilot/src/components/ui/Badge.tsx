import { clsx } from 'clsx'

interface BadgeProps {
  variant?: 'default' | 'live' | 'stale' | 'offline' | 'demo' | 'act' | 'wait' | 'nothing' | 'success' | 'warning' | 'error'
  size?: 'sm' | 'md' | 'lg'
  children: React.ReactNode
  className?: string
}

const variantStyles = {
  default: 'bg-zinc-600/20 text-zinc-400 border border-zinc-600/30',
  live: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
  stale: 'bg-amber-500/15 text-amber-400 border border-amber-500/30',
  offline: 'bg-red-500/15 text-red-400 border border-red-500/30',
  demo: 'bg-purple-500/15 text-purple-400 border border-purple-500/30',
  act: 'bg-cyan-500/20 text-cyan-300 border border-cyan-400/40',
  wait: 'bg-amber-500/15 text-amber-400 border border-amber-500/30',
  nothing: 'bg-zinc-600/20 text-zinc-400 border border-zinc-600/30',
  success: 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30',
  warning: 'bg-amber-500/15 text-amber-400 border border-amber-500/30',
  error: 'bg-red-500/15 text-red-400 border border-red-500/30',
}

const sizeStyles = {
  sm: 'px-1.5 py-0.5 text-[10px] tracking-wide',
  md: 'px-2 py-0.5 text-[11px] tracking-wide',
  lg: 'px-2.5 py-1 text-xs tracking-wide',
}

export function Badge({ variant = 'default', size = 'md', children, className }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded-sm font-semibold uppercase',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
    >
      {children}
    </span>
  )
}

// Specialized badges with refined dot indicators
export function LiveBadge({ compact }: { compact?: boolean } = {}) {
  return (
    <Badge variant="live" size="sm">
      <span className="relative flex h-1.5 w-1.5">
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
        <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-400" />
      </span>
      {!compact && 'LIVE'}
    </Badge>
  )
}

export function StaleBadge({ compact }: { compact?: boolean } = {}) {
  return (
    <Badge variant="stale" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
      {!compact && 'STALE'}
    </Badge>
  )
}

export function DemoBadge({ compact }: { compact?: boolean } = {}) {
  return (
    <Badge variant="demo" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-purple-400 animate-pulse" />
      {!compact && 'DEMO'}
    </Badge>
  )
}

export function OfflineBadge({ compact }: { compact?: boolean } = {}) {
  return (
    <Badge variant="offline" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
      {!compact && 'OFFLINE'}
    </Badge>
  )
}

// As-of timestamp chip for data freshness
export function AsOfChip({ time, className }: { time: Date | string | null; className?: string }) {
  if (!time) return null
  
  const date = typeof time === 'string' ? new Date(time) : time
  const now = new Date()
  const diffSec = Math.floor((now.getTime() - date.getTime()) / 1000)
  
  let label: string
  if (diffSec < 5) label = 'now'
  else if (diffSec < 60) label = `${diffSec}s`
  else if (diffSec < 3600) label = `${Math.floor(diffSec / 60)}m`
  else label = date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
  
  return (
    <span className={clsx('as-of-chip', className)}>
      <span className="opacity-60">as-of</span>
      <span className="mono-nums">{label}</span>
    </span>
  )
}
