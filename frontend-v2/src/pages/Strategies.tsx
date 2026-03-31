import { useState, useEffect } from 'react'
import { api } from '../lib/api'
import { useToast } from '../components/Toast'
import ui from '../components/ui.module.css'
import s from './Strategies.module.css'

interface Strategy { id: string; name: string; description?: string; trades: number; win_rate: number; pf: number; net_pnl: number; active: boolean }

const GRADIENTS: Record<string, string> = {
  classic_gap_fade: 'linear-gradient(135deg, #0d1b2a 0%, #00c853 100%)',
  exhaustion_gap_fade: 'linear-gradient(135deg, #0a1a1a 0%, #00FFBB 100%)',
  vwap_gap_fade: 'linear-gradient(135deg, #0d2a1a 0%, #06b6d4 100%)',
  confluence_gap: 'linear-gradient(135deg, #1a0d2a 0%, #a855f7 100%)',
  minervini_trend: 'linear-gradient(135deg, #2a1a0d 0%, #f97316 100%)',
  gap_continuation: 'linear-gradient(135deg, #1a0d2a 0%, #8b5cf6 100%)',
  gap_bounce: 'linear-gradient(135deg, #2a0d0d 0%, #ff3b5c 100%)',
}

export default function Strategies() {
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [activeId, setActiveId] = useState('')
  const [intradayStrats, setIntradayStrats] = useState<Record<string, unknown>[]>([])
  const [perf, setPerf] = useState<Record<string, unknown> | null>(null)
  const { showToast } = useToast()

  useEffect(() => {
    Promise.all([api.strategies(), api.trades()]).then(([reg, trades]) => {
      setActiveId(reg.active)
      const statsMap = new Map<string, { wins: number; losses: number; gp: number; gl: number }>()
      for (const t of trades) {
        const sid = t.strategy_id || 'unknown'
        if (!statsMap.has(sid)) statsMap.set(sid, { wins: 0, losses: 0, gp: 0, gl: 0 })
        const st = statsMap.get(sid)!
        if (t.pnl > 0) { st.wins++; st.gp += t.pnl } else { st.losses++; st.gl += Math.abs(t.pnl) }
      }
      const result: Strategy[] = []
      const seen = new Set<string>()
      for (const rs of reg.strategies) {
        seen.add(rs.id)
        const st = statsMap.get(rs.id)
        const total = st ? st.wins + st.losses : 0
        result.push({ id: rs.id, name: rs.name, description: rs.description, trades: total,
          win_rate: total > 0 ? st!.wins / total : 0,
          pf: st && st.gl > 0 ? st.gp / st.gl : st && st.gp > 0 ? 99 : 0,
          net_pnl: st ? st.gp - st.gl : 0, active: rs.id === reg.active })
      }
      const NAME_MAP: Record<string, string> = {
        'unknown': 'Classic Gap Fade (Legacy)',
        '': 'Classic Gap Fade (Legacy)',
      }
      for (const [sid, st] of statsMap) {
        if (seen.has(sid)) continue
        const total = st.wins + st.losses
        result.push({ id: sid, name: NAME_MAP[sid] ?? sid.replace(/_/g, ' '), trades: total,
          win_rate: total > 0 ? st.wins / total : 0,
          pf: st.gl > 0 ? st.gp / st.gl : st.gp > 0 ? 99 : 0,
          net_pnl: st.gp - st.gl, active: sid === reg.active })
      }
      result.sort((a, b) => (a.active ? -1 : 1) - (b.active ? -1 : 1) || b.net_pnl - a.net_pnl)
      setStrategies(result)
    }).catch(() => {})

    fetch('/api/intraday/strategies').then(r => r.json()).then(d => setIntradayStrats(d.strategies ?? d ?? [])).catch(() => {})
    fetch('/api/intraday/performance').then(r => r.json()).then(setPerf).catch(() => {})
  }, [])

  const switchStrategy = async (id: string) => {
    try {
      await fetch('/api/strategy/switch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ strategy_id: id }) })
      setActiveId(id)
      setStrategies(prev => prev.map(st => ({ ...st, active: st.id === id })))
      showToast(`Switched to ${id}`, 'success')
    } catch { showToast('Failed to switch strategy', 'error') }
  }

  return (
    <div className={s.page}>
      {/* Gap Fade Strategies */}
      <div className={s.sectionTitle}>Gap Fade Strategies</div>
      <div className={s.gallery}>
        {strategies.map(st => (
          <div key={st.id} className={`${s.card} ${st.active ? s.cardActive : ''}`}
            style={{ background: GRADIENTS[st.id] ?? 'linear-gradient(135deg, #1a1a1a 0%, #333 100%)' }}
            onClick={() => switchStrategy(st.id)}>
            <div className={s.cardContent}>
              <div className={s.cardTop}>
                <div className={s.cardName}>{st.name.toUpperCase()}</div>
                {st.active && <span className={`${ui.badge} ${ui.badgeLive} ${ui.badgeSm}`}>ACTIVE</span>}
              </div>
              {st.description && <div className={s.cardDesc}>{st.description}</div>}
              <div className={s.cardStats}>
                <div className={s.cardStat}><span className={s.statLabel}>Trades</span><span className={s.statValue}>{st.trades}</span></div>
                <div className={s.cardStat}><span className={s.statLabel}>Win Rate</span><span className={s.statValue}>{(st.win_rate * 100).toFixed(1)}%</span></div>
                <div className={s.cardStat}><span className={s.statLabel}>PF</span><span className={s.statValue}>{st.pf >= 99 ? '∞' : st.pf.toFixed(2)}</span></div>
                <div className={s.cardStat}><span className={s.statLabel}>Net P&L</span><span className={`${s.statValue} ${st.net_pnl >= 0 ? s.positive : s.negative}`}>{st.net_pnl >= 0 ? '+' : ''}${st.net_pnl.toFixed(0)}</span></div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Intraday Strategies */}
      {intradayStrats.length > 0 && (
        <>
          <div className={s.sectionTitle} style={{ marginTop: 24 }}>Intraday Strategies</div>
          <div className={s.intradayList}>
            {intradayStrats.map((ist, i) => (
              <div key={i} className={s.intradayRow}>
                <div>
                  <div className={s.intradayName}>{String(ist.name ?? ist.id ?? '')}</div>
                  <div className={s.intradayDesc}>{String(ist.description ?? '')}</div>
                </div>
                <span className={`${ui.badge} ${ist.enabled ? ui.badgeAccent : ui.badgeMuted} ${ui.badgeSm}`}>
                  {ist.enabled ? 'ON' : 'OFF'}
                </span>
              </div>
            ))}
          </div>
        </>
      )}

      {/* Performance */}
      {perf && (
        <>
          <div className={s.sectionTitle} style={{ marginTop: 24 }}>Performance</div>
          <div className={s.perfGrid}>
            {Object.entries(perf).filter(([, v]) => typeof v === 'number').slice(0, 6).map(([k, v]) => (
              <div key={k} className={s.perfItem}>
                <span className={s.perfLabel}>{k.replace(/_/g, ' ')}</span>
                <span className={s.perfValue}>{typeof v === 'number' ? (v as number).toFixed(2) : String(v)}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
