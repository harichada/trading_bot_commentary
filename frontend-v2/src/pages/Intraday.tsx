import { useState, useEffect } from 'react'
import s from './Intraday.module.css'

function toArray(v: unknown): Record<string, unknown>[] {
  if (Array.isArray(v)) return v
  if (v && typeof v === 'object' && !('error' in (v as Record<string, unknown>))) return Object.values(v)
  return []
}

export default function Intraday() {
  const [state, setState] = useState<Record<string, unknown> | null>(null)
  const [strategies, setStrategies] = useState<Record<string, unknown>[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.allSettled([
      fetch('/api/intraday/state').then(r => r.json()),
      fetch('/api/intraday/strategies').then(r => r.json()),
    ]).then(([stRes, strRes]) => {
      if (stRes.status === 'fulfilled' && stRes.value && !stRes.value.error) setState(stRes.value)
      if (strRes.status === 'fulfilled') {
        const d = strRes.value
        setStrategies(toArray(d?.strategies ?? d))
      }
      setLoading(false)
    })
  }, [])

  const signals = toArray(state?.signals)
  const sessionPnl = Number(state?.session_pnl ?? 0)
  const trades = Number(state?.trades_today ?? 0)

  if (loading) return <div className={s.page}><div className={s.empty}>Loading intraday data...</div></div>

  return (
    <div className={s.page}>
      <div className={s.summary}>
        <div className={s.stat}>
          <span className={s.statLabel}>Session P&L</span>
          <span className={`${s.statValue} ${sessionPnl >= 0 ? s.up : s.down}`}>
            {sessionPnl >= 0 ? '+' : ''}${sessionPnl.toFixed(2)}
          </span>
        </div>
        <div className={s.stat}>
          <span className={s.statLabel}>Trades Today</span>
          <span className={s.statValue}>{trades}</span>
        </div>
        <div className={s.stat}>
          <span className={s.statLabel}>Strategies</span>
          <span className={s.statValue}>{strategies.length}</span>
        </div>
      </div>

      <div className={s.section}>
        <div className={s.sectionTitle}>Active Strategies</div>
        {strategies.length === 0 ? (
          <div className={s.empty}>No intraday strategies configured</div>
        ) : (
          <div className={s.stratList}>
            {strategies.map((st, i) => (
              <div key={i} className={s.stratRow}>
                <span className={s.stratName}>{String(st.name ?? st.id ?? `Strategy ${i+1}`)}</span>
                <span className={s.stratMeta}>{String(st.description ?? '')}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className={s.section}>
        <div className={s.sectionTitle}>Recent Signals</div>
        {signals.length === 0 ? (
          <div className={s.empty}>No signals — market may be closed</div>
        ) : (
          <div className={s.tableWrap}>
            <table className={s.table}>
              <thead><tr><th>Time</th><th>Symbol</th><th>Strategy</th><th>Direction</th><th>Signal</th></tr></thead>
              <tbody>{signals.slice(0, 20).map((sig, i) => (
                <tr key={i}>
                  <td className={s.mono}>{String(sig.time ?? '')}</td>
                  <td className={s.ticker}>{String(sig.symbol ?? '')}</td>
                  <td>{String(sig.strategy ?? '')}</td>
                  <td className={String(sig.direction) === 'long' ? s.up : s.down}>{String(sig.direction ?? '').toUpperCase()}</td>
                  <td>{String(sig.signal ?? '')}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
