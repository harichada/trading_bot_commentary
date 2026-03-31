import { useState } from 'react'
import { useToast } from '../components/Toast'
import s from './WalkForward.module.css'

export default function WalkForward() {
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<Record<string, unknown>[] | null>(null)
  const [summary, setSummary] = useState<Record<string, unknown> | null>(null)
  const [progress, setProgress] = useState('')
  const { showToast } = useToast()
  const [form, setForm] = useState({
    universe: 'study', start_date: '2022-01-01', end_date: '2025-06-30',
    train_days: 504, test_days: 252, step_days: 126, capital: 25000,
    grid_gap: '0.03,0.04,0.05', grid_vol: '5,8,12', grid_stop: '0.02,0.03,0.04', grid_risk: '0.01,0.015,0.02',
  })

  const gridCount = form.grid_gap.split(',').length * form.grid_vol.split(',').length * form.grid_stop.split(',').length * form.grid_risk.split(',').length

  const run = async () => {
    setRunning(true); setResults(null); setSummary(null)
    try {
      await fetch('/api/backtest/walkforward', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) })
      const poll = setInterval(async () => {
        const st = await fetch('/api/backtest/walkforward/status').then(r => r.json())
        setProgress(st.message ?? st.status ?? '')
        if (st.status === 'complete' || st.status === 'idle') {
          clearInterval(poll); setRunning(false)
          const res = await fetch('/api/backtest/walkforward/results').then(r => r.json())
          setResults(res.windows ?? res.results ?? [])
          setSummary(res.summary ?? res)
          showToast('Walk-forward complete', 'success')
        }
        if (st.status === 'error') { clearInterval(poll); setRunning(false); showToast('Walk-forward failed', 'error') }
      }, 3000)
    } catch { setRunning(false); showToast('Failed to start', 'error') }
  }

  const cancel = () => { fetch('/api/backtest/walkforward/cancel', { method: 'POST' }); setRunning(false) }

  return (
    <div className={s.page}>
      {/* Controls */}
      <div className={s.controls}>
        <div className={s.field}><label>Universe</label>
          <select value={form.universe} onChange={e => setForm({...form, universe: e.target.value})}>
            <option value="study">Study (167)</option><option value="alpaca">All Tradeable</option>
          </select></div>
        <div className={s.field}><label>Start</label><input type="date" value={form.start_date} onChange={e => setForm({...form, start_date: e.target.value})} /></div>
        <div className={s.field}><label>End</label><input type="date" value={form.end_date} onChange={e => setForm({...form, end_date: e.target.value})} /></div>
        <div className={s.field}><label>Train Days</label><input type="number" value={form.train_days} onChange={e => setForm({...form, train_days: +e.target.value})} /></div>
        <div className={s.field}><label>Test Days</label><input type="number" value={form.test_days} onChange={e => setForm({...form, test_days: +e.target.value})} /></div>
        <div className={s.field}><label>Step Days</label><input type="number" value={form.step_days} onChange={e => setForm({...form, step_days: +e.target.value})} /></div>
        <div className={s.field}><label>Capital</label><input type="number" value={form.capital} onChange={e => setForm({...form, capital: +e.target.value})} /></div>
      </div>

      {/* Parameter Grid */}
      <details className={s.details}>
        <summary>Parameter Grid ({gridCount} combinations)</summary>
        <div className={s.gridFields}>
          <div className={s.field}><label>Gap % values</label><input value={form.grid_gap} onChange={e => setForm({...form, grid_gap: e.target.value})} /></div>
          <div className={s.field}><label>Vol Max values</label><input value={form.grid_vol} onChange={e => setForm({...form, grid_vol: e.target.value})} /></div>
          <div className={s.field}><label>Stop % values</label><input value={form.grid_stop} onChange={e => setForm({...form, grid_stop: e.target.value})} /></div>
          <div className={s.field}><label>Risk % values</label><input value={form.grid_risk} onChange={e => setForm({...form, grid_risk: e.target.value})} /></div>
        </div>
      </details>

      {/* Actions */}
      <div className={s.actions}>
        <button className={s.btnRun} onClick={run} disabled={running}>{running ? 'Running...' : '▶ Run Walk-Forward'}</button>
        {running && <button className={s.btnCancel} onClick={cancel}>Cancel</button>}
        {running && <span className={s.progressText}>{progress}</span>}
      </div>

      {/* Results */}
      {results && results.length > 0 && (
        <div className={s.resultsSection}>
          {summary && (
            <div className={s.summaryGrid}>
              {['oos_sharpe', 'oos_return', 'oos_pf', 'oos_win_rate', 'wf_efficiency', 'total_oos_trades'].map(k => (
                <div key={k} className={s.summaryItem}>
                  <span className={s.summaryLabel}>{k.replace(/_/g, ' ')}</span>
                  <span className={s.summaryValue}>{typeof summary[k] === 'number' ? (summary[k] as number).toFixed(2) : String(summary[k] ?? '—')}</span>
                </div>
              ))}
            </div>
          )}
          <div className={s.tableWrap}>
            <table className={s.table}>
              <thead><tr><th>#</th><th>Train Period</th><th>Test Period</th><th>IS Return</th><th>OOS Return</th><th>Efficiency</th></tr></thead>
              <tbody>{results.map((w, i) => {
                const oos = Number(w.oos_return ?? 0), is_ = Number(w.is_return ?? 0)
                const eff = is_ !== 0 ? (oos / is_ * 100) : 0
                return (<tr key={i}>
                  <td>{i+1}</td><td className={s.mono}>{String(w.train_start ?? '')} → {String(w.train_end ?? '')}</td>
                  <td className={s.mono}>{String(w.test_start ?? '')} → {String(w.test_end ?? '')}</td>
                  <td className={is_ >= 0 ? s.up : s.down}>{is_.toFixed(2)}%</td>
                  <td className={oos >= 0 ? s.up : s.down}>{oos.toFixed(2)}%</td>
                  <td className={eff >= 70 ? s.up : eff >= 50 ? s.warn : s.down}>{eff.toFixed(0)}%</td>
                </tr>)})}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!running && !results && <div className={s.empty}>Configure walk-forward parameters and run</div>}
    </div>
  )
}
