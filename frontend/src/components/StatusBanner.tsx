import { type ApiError } from '../api/client'

interface StatusBannerProps {
  label: string
  error: ApiError | null
  wsConnected: boolean
  onRetry?: () => void
}

export default function StatusBanner({ label, error, wsConnected, onRetry }: StatusBannerProps) {
  if (error?.isAuth) {
    return (
      <div className="mb-4 px-4 py-2 rounded-lg bg-yellow/10 border border-yellow/30 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-yellow" />
          <span className="text-xs font-semibold text-yellow">{label}: Authentication Required</span>
          <span className="text-[10px] text-text-muted">Log in to access live data</span>
        </div>
        <a href="/login" className="text-[10px] font-bold text-yellow hover:text-yellow/80 transition-colors">
          LOGIN
        </a>
      </div>
    )
  }

  if (error?.isOffline) {
    return (
      <div className="mb-4 px-4 py-2 rounded-lg bg-red/10 border border-red/30 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-red" />
          <span className="text-xs font-semibold text-red">{label}: Offline</span>
          <span className="text-[10px] text-text-muted">Service unreachable</span>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-[10px] font-bold text-red hover:text-red/80 transition-colors"
          >
            RETRY
          </button>
        )}
      </div>
    )
  }

  if (!wsConnected) {
    return (
      <div className="mb-4 px-4 py-2 rounded-lg bg-yellow/10 border border-yellow/30 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-yellow animate-pulse" />
        <span className="text-xs font-semibold text-yellow">{label}: Reconnecting...</span>
        <span className="text-[10px] text-text-muted">WebSocket disconnected, auto-retrying every 5s</span>
      </div>
    )
  }

  return null
}
