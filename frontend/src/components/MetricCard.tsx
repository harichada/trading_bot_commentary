interface MetricCardProps {
  label: string
  value: string
  subValue?: string
  trend?: 'up' | 'down' | 'neutral'
  accentColor?: string
}

export default function MetricCard({ label, value, subValue, trend, accentColor }: MetricCardProps) {
  const trendColor = trend === 'up' ? 'text-green' : trend === 'down' ? 'text-red' : 'text-text-secondary'
  const borderColor = accentColor ?? (trend === 'up' ? '#22c55e' : trend === 'down' ? '#ef4444' : '#1a1a1a')

  return (
    <div
      className="bg-bg-card rounded-lg p-4 border border-border"
      style={{ borderBottomColor: borderColor, borderBottomWidth: '2px' }}
    >
      <p className="text-text-secondary text-xs uppercase tracking-wider mb-1">{label}</p>
      <p className={`font-mono text-xl font-semibold ${trendColor}`}>{value}</p>
      {subValue && (
        <p className={`font-mono text-xs mt-1 ${trendColor}`}>{subValue}</p>
      )}
    </div>
  )
}
