import { useTraderState } from '../hooks/useTraderState'
import { api } from '../lib/api'
import ui from '../components/ui.module.css'
import s from './Scanner.module.css'

const CATALYST_COLORS: Record<string, string> = {
  earnings: ui.badgeWarning, earn: ui.badgeWarning,
  fda: ui.badgePurple, ma: ui.badgeLoss, 'm&a': ui.badgeLoss,
  offering: ui.badgeAccent, offer: ui.badgeAccent,
  upgrade: ui.badgeProfit, upg: ui.badgeProfit,
  downgrade: ui.badgeLoss, dng: ui.badgeLoss,
  noise: ui.badgeMuted,
}

function scoreBadge(score: number) {
  if (score >= 90) return { cls: ui.badgeAccent, label: 'A+' }
  if (score >= 75) return { cls: ui.badgeProfit, label: 'B+' }
  if (score >= 60) return { cls: ui.badgeWarning, label: 'C' }
  return { cls: ui.badgeMuted, label: '—' }
}

export default function Scanner() {
  const { state } = useTraderState(5000)
  const candidates = state?.candidates ?? []

  return (
    <div className={s.page}>
      <div className={s.header}>
        <span className={s.title}>Scanner</span>
        <span className={s.meta}>{candidates.length} candidates</span>
        <button className={s.scanBtn} onClick={() => api.scan()}>Run Scan</button>
      </div>

      <div className={s.tableWrap}>
        <table className={s.table}>
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Dir</th>
              <th>Gap %</th>
              <th>Prev Close</th>
              <th>Current</th>
              <th>Vol Ratio</th>
              <th>Avg Vol</th>
              <th>Short</th>
              <th>ETB</th>
              <th>Catalyst</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {candidates.length === 0 && (
              <tr><td colSpan={11} className={s.empty}>No candidates — run a scan or wait for pre-market</td></tr>
            )}
            {candidates.map((c: Record<string, unknown>) => {
              const gap = Number(c.gap_pct ?? 0)
              const score = Number(c.score ?? 0)
              const { cls, label } = scoreBadge(score)
              const dir = String(c.direction ?? '')
              const vol = Number(c.vol_ratio ?? 0)
              const catalyst = String(c.catalyst ?? '')
              const catCls = CATALYST_COLORS[catalyst.toLowerCase()] ?? ui.badgeMuted
              const shortable = c.shortable !== false && c.shortable !== 0
              const etb = c.easy_to_borrow === true || c.etb === true

              return (
                <tr key={String(c.symbol)}>
                  <td className={s.ticker}>{String(c.symbol)}</td>
                  <td><span className={`${ui.badge} ${dir === 'long' ? ui.badgeProfit : ui.badgeLoss} ${ui.badgeSm}`}>{dir.toUpperCase()}</span></td>
                  <td className={`${s.mono} ${gap > 0 ? s.up : s.down}`}>{(gap * 100).toFixed(1)}%</td>
                  <td className={s.mono}>${Number(c.prev_close ?? 0).toFixed(2)}</td>
                  <td className={s.mono}>${Number(c.premarket_price ?? 0).toFixed(2)}</td>
                  <td className={`${s.mono} ${vol > 3 ? s.up : vol > 1.5 ? '' : s.muted}`}>{vol.toFixed(1)}x</td>
                  <td className={`${s.mono} ${s.muted}`}>{Number(c.avg_volume ?? 0) > 1000 ? `${(Number(c.avg_volume ?? 0) / 1000).toFixed(0)}K` : String(c.avg_volume ?? '—')}</td>
                  <td>{shortable ? <span className={s.up}>✓</span> : <span className={s.down}>✗</span>}</td>
                  <td>{etb ? <span className={s.muted}>✓</span> : <span className={s.muted}>—</span>}</td>
                  <td>{catalyst ? <span className={`${ui.badge} ${catCls} ${ui.badgeSm}`}>{catalyst.toUpperCase()}</span> : <span className={s.muted}>—</span>}</td>
                  <td>
                    <span className={s.scoreBar}>
                      <span className={s.scoreFill} style={{ width: `${Math.min(100, score)}%`, background: score >= 90 ? '#00FFBB' : score >= 75 ? '#4ade80' : score >= 60 ? '#F5A623' : '#555' }} />
                    </span>
                    <span className={s.scoreNum}>{score.toFixed(0)}</span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
