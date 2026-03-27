interface StrategyCardProps {
  name: string
  market: string
  status: 'ACTIVE' | 'PAUSED' | 'STOPPED'
  dailyPnl: number
  winRate: number
  equityCurve: number[]
  onSettings?: () => void
}

export default function StrategyCard({ name, market, status, dailyPnl, winRate, equityCurve, onSettings }: StrategyCardProps) {
  const statusColors = {
    ACTIVE: 'bg-green/20 text-green',
    PAUSED: 'bg-yellow/20 text-yellow',
    STOPPED: 'bg-red/20 text-red',
  }

  const maxVal = Math.max(...equityCurve, 1)
  const minVal = Math.min(...equityCurve, 0)
  const range = maxVal - minVal || 1

  const pathPoints = equityCurve
    .map((v, i) => {
      const x = (i / Math.max(equityCurve.length - 1, 1)) * 160
      const y = 60 - ((v - minVal) / range) * 50
      return `${x},${y}`
    })
    .join(' ')

  return (
    <div className="bg-bg-card rounded-lg border border-border p-4 hover:border-border-light transition-colors">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-text-primary">{name}</h3>
        <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${statusColors[status]}`}>
          {status}
        </span>
      </div>
      <p className="text-xs text-text-secondary mb-3">{market}</p>

      <div className="flex gap-4 mb-3">
        <div>
          <p className="text-[10px] text-text-muted uppercase">Daily P&L</p>
          <p className={`font-mono text-sm font-semibold ${dailyPnl >= 0 ? 'text-green' : 'text-red'}`}>
            {dailyPnl >= 0 ? '+' : ''}${dailyPnl.toLocaleString('en-US', { minimumFractionDigits: 2 })}
          </p>
        </div>
        <div>
          <p className="text-[10px] text-text-muted uppercase">Win Rate</p>
          <p className="font-mono text-sm font-semibold text-text-primary">{winRate.toFixed(1)}%</p>
        </div>
      </div>

      <div className="h-16 mb-3">
        <svg viewBox="0 0 160 65" className="w-full h-full">
          <polyline
            points={pathPoints}
            fill="none"
            stroke="#22c55e"
            strokeWidth="1.5"
          />
        </svg>
      </div>

      <div className="flex gap-2">
        <button
          onClick={onSettings}
          className="flex-1 text-xs py-1.5 rounded bg-bg-card-hover text-text-secondary hover:text-text-primary transition-colors"
        >
          SETTINGS
        </button>
      </div>
    </div>
  )
}
