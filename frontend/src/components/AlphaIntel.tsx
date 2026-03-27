interface Signal {
  type: string
  title: string
  description: string
  time: string
}

interface AlphaIntelProps {
  signals: Signal[]
}

export default function AlphaIntel({ signals }: AlphaIntelProps) {
  const getStyles = (type: string) => {
    const t = type.toUpperCase()
    if (t.includes('BUY') || t.includes('LONG') || t.includes('BULLISH')) {
      return { border: 'border-l-green bg-green/5 text-green', label: 'bg-green/20 text-green' }
    }
    if (t.includes('SELL') || t.includes('SHORT') || t.includes('REDUCE') || t.includes('BEARISH')) {
      return { border: 'border-l-red bg-red/5 text-red', label: 'bg-red/20 text-red' }
    }
    return { border: 'border-l-yellow bg-yellow/5 text-yellow', label: 'bg-yellow/20 text-yellow' }
  }

  return (
    <div className="bg-bg-card rounded-lg border border-border p-4">
      <div className="flex items-center gap-2 mb-4">
        <span className="text-purple">&#9670;</span>
        <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider">Activity Feed</h3>
      </div>

      {signals.length === 0 ? (
        <p className="text-xs text-text-muted">No signals yet</p>
      ) : (
        <div className="space-y-3">
          {signals.map((signal, i) => {
            const styles = getStyles(signal.type)
            return (
              <div key={i} className={`border-l-2 rounded-r-lg p-3 ${styles.border}`}>
                <div className="flex items-center justify-between mb-1">
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${styles.label}`}>
                    {signal.type}
                  </span>
                  <span className="text-[10px] text-text-muted">{signal.time}</span>
                </div>
                <p className="text-xs font-semibold text-text-primary mt-1">{signal.title}</p>
                <p className="text-[11px] text-text-secondary mt-1 leading-relaxed">{signal.description}</p>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
