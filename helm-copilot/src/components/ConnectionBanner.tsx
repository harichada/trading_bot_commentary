import { WifiOff, RefreshCw } from 'lucide-react'
import { clsx } from 'clsx'
import { DemoBadge, OfflineBadge } from './ui/Badge'
import { Button } from './ui/Button'

interface ConnectionBannerProps {
  state: 'connecting' | 'connected' | 'disconnected' | 'offline' | 'demo'
  isDemo: boolean
  lastError?: string | null
  onRetry?: () => void
}

export function ConnectionBanner({ state, isDemo, lastError, onRetry }: ConnectionBannerProps) {
  if (state === 'connected' && !isDemo) {
    return null
  }

  const isOffline = state === 'offline' || state === 'disconnected'

  return (
    <div
      className={clsx(
        'w-full px-4 py-2 flex items-center justify-between gap-3',
        isDemo && 'bg-purple-500/10 border-b border-purple-500/20',
        isOffline && !isDemo && 'bg-red-500/10 border-b border-red-500/20'
      )}
    >
      <div className="flex items-center gap-3">
        {isOffline && !isDemo && (
          <>
            <WifiOff className="w-4 h-4 text-red-400" />
            <span className="text-sm text-red-400">
              Unable to connect to trading bot API
            </span>
            <OfflineBadge />
          </>
        )}
        {isDemo && (
          <>
            <span className="text-sm text-purple-400">
              Running in demo mode — data shown is simulated
            </span>
            <DemoBadge />
          </>
        )}
        {state === 'connecting' && !isDemo && (
          <span className="text-sm text-amber-400 flex items-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin" />
            Connecting to trading bot...
          </span>
        )}
      </div>
      
      <div className="flex items-center gap-2">
        {lastError && (
          <span className="text-xs text-zinc-500 hidden sm:inline">
            {lastError}
          </span>
        )}
        {(isOffline || isDemo) && onRetry && (
          <Button variant="ghost" size="sm" onClick={onRetry}>
            <RefreshCw className="w-3 h-3" />
            Retry
          </Button>
        )}
      </div>
    </div>
  )
}
