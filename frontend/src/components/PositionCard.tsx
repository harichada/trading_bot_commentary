interface PositionCardProps {
  symbol: string
  shares: number
  avgPrice: number
  currentPrice: number
  pnl: number
  pnlPercent: number
}

export default function PositionCard({ symbol, shares, avgPrice, currentPrice, pnl, pnlPercent }: PositionCardProps) {
  const isPositive = pnl >= 0

  return (
    <div className="flex items-center justify-between py-3 px-4 border-b border-border hover:bg-bg-card-hover transition-colors">
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full bg-bg-card-hover flex items-center justify-center text-xs font-semibold text-cyan">
          {symbol.charAt(0)}
        </div>
        <div>
          <p className="text-sm font-semibold text-text-primary">{symbol}</p>
          <p className="text-xs text-text-secondary font-mono">
            {shares.toLocaleString()} @ ${avgPrice.toFixed(2)}
          </p>
        </div>
      </div>
      <div className="text-right">
        <p className={`font-mono text-sm font-semibold ${isPositive ? 'text-green' : 'text-red'}`}>
          {isPositive ? '+' : ''}{pnl >= 0 ? '$' : '-$'}{Math.abs(pnl).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </p>
        <p className={`font-mono text-xs ${isPositive ? 'text-green' : 'text-red'}`}>
          {isPositive ? '+' : ''}{pnlPercent.toFixed(2)}%
        </p>
      </div>
    </div>
  )
}
