import { clsx } from 'clsx'

interface BadgeProps {
  variant?: 'default' | 'live' | 'stale' | 'offline' | 'demo' | 'act' | 'wait' | 'nothing' | 'success' | 'warning' | 'error'
  size?: 'sm' | 'md'
  children: React.ReactNode
  className?: string
}

const variantStyles = {
  default: 'bg-zinc-500/20 text-zinc-400 border border-zinc-500/30',
  live: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
  stale: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
  offline: 'bg-red-500/20 text-red-400 border border-red-500/30',
  demo: 'bg-purple-500/20 text-purple-400 border border-purple-500/30',
  act: 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30',
  wait: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
  nothing: 'bg-zinc-500/20 text-zinc-400 border border-zinc-500/30',
  success: 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30',
  warning: 'bg-amber-500/20 text-amber-400 border border-amber-500/30',
  error: 'bg-red-500/20 text-red-400 border border-red-500/30',
}

const sizeStyles = {
  sm: 'px-1.5 py-0.5 text-[10px]',
  md: 'px-2 py-0.5 text-xs',
}

export function Badge({ variant = 'default', size = 'md', children, className }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded font-medium',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
    >
      {children}
    </span>
  )
}

// Specialized badges
export function LiveBadge() {
  return (
    <Badge variant="live" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
      LIVE
    </Badge>
  )
}

export function StaleBadge() {
  return (
    <Badge variant="stale" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
      STALE
    </Badge>
  )
}

export function DemoBadge() {
  return (
    <Badge variant="demo" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
      DEMO
    </Badge>
  )
}

export function OfflineBadge() {
  return (
    <Badge variant="offline" size="sm">
      <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
      OFFLINE
    </Badge>
  )
}
