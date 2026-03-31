interface Props {
  winRate: number  // 0-100
  size?: number
}

export default function WinRateGauge({ winRate, size = 100 }: Props) {
  const r = size * 0.375
  const circumference = 2 * Math.PI * r
  const progress = (winRate / 100) * circumference
  const color = winRate >= 55 ? '#00FFBB' : winRate >= 45 ? '#F5A623' : '#FF4D6A'
  const cx = size / 2
  const cy = size / 2

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block' }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={8} />
      <circle
        cx={cx} cy={cy} r={r}
        fill="none"
        stroke={color}
        strokeWidth={8}
        strokeDasharray={`${progress.toFixed(2)} ${circumference.toFixed(2)}`}
        strokeLinecap="round"
        transform={`rotate(-90 ${cx} ${cy})`}
        style={{ transition: 'stroke-dasharray 0.8s cubic-bezier(0.4,0,0.2,1)' }}
      />
      <text x={cx} y={cy - 3} textAnchor="middle" fontFamily="JetBrains Mono, monospace" fontSize={14} fontWeight={700} fill="#F0F0F0">
        {winRate.toFixed(0)}%
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fontFamily="Inter, sans-serif" fontSize={9} fontWeight={500} fill="#606060" letterSpacing="0.1em">
        WIN
      </text>
    </svg>
  )
}
