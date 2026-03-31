import { useState } from 'react'
import ui from '../components/ui.module.css'
import s from './WalkForward.module.css'

export default function IntradayWF() {
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<Record<string, unknown>[] | null>(null)
  const [form, setForm] = useState({ in_sample: 60, out_sample: 15, step: 15 })

  const run = async () => {
    setRunning(true)
    try {
      await fetch('/api/backtest/intraday-walkforward', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form),
      })
      const poll = setInterval(async () => {
        const st = await fetch('/api/backtest/intraday-walkforward/status').then(r => r.json())
        if (st.status === 'complete' || st.status === 'idle') {
          clearInterval(poll)
          setRunning(false)
          const res = await fetch('/api/backtest/intraday-walkforward/results').then(r => r.json())
          setResults(res.windows ?? res.results ?? [])
        }
        if (st.status === 'error') { clearInterval(poll); setRunning(false) }
      }, 3000)
    } catch { setRunning(false) }
  }

  return (
    <div className={s.page}>
      <div className={s.controls}>
        <div className={s.field}><label>In-Sample (days)</label>
          <input type="number" className={ui.input} value={form.in_sample} onChange={e => setForm({...form, in_sample: +e.target.value})} style={{width:100}} /></div>
        <div className={s.field}><label>Out-Sample (days)</label>
          <input type="number" className={ui.input} value={form.out_sample} onChange={e => setForm({...form, out_sample: +e.target.value})} style={{width:100}} /></div>
        <div className={s.field}><label>Step (days)</label>
          <input type="number" className={ui.input} value={form.step} onChange={e => setForm({...form, step: +e.target.value})} style={{width:100}} /></div>
        <button className={`${ui.btn} ${ui.btnPrimary}`} onClick={run} disabled={running}>{running ? 'Running...' : '▶ Run'}</button>
      </div>
      {results && results.length > 0 && (
        <div className={s.tableWrap}>
          <table className={s.table}>
            <thead><tr><th>#</th><th>Train Period</th><th>Test Period</th><th>IS Return</th><th>OOS Return</th></tr></thead>
            <tbody>{results.map((w: Record<string, unknown>, i: number) => {
              const oos = Number(w.oos_return ?? 0), is_ = Number(w.is_return ?? 0)
              return (<tr key={i}>
                <td>{i+1}</td>
                <td className={s.mono}>{String(w.train_start ?? '')} → {String(w.train_end ?? '')}</td>
                <td className={s.mono}>{String(w.test_start ?? '')} → {String(w.test_end ?? '')}</td>
                <td className={is_ >= 0 ? s.up : s.down}>{is_.toFixed(2)}%</td>
                <td className={oos >= 0 ? s.up : s.down}>{oos.toFixed(2)}%</td>
              </tr>)})}
            </tbody>
          </table>
        </div>
      )}
      {!running && !results && <div className={s.empty}>Configure intraday walk-forward parameters and run</div>}
    </div>
  )
}
