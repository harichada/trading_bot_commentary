import { useState, useEffect } from 'react'
import { useToast } from '../components/Toast'
import s from './Database.module.css'

export default function Database() {
  const [stats, setStats] = useState<Record<string, unknown> | null>(null)
  const [building, setBuilding] = useState(false)
  const [progress, setProgress] = useState<{ msg: string; pct: number }>({ msg: '', pct: 0 })
  const [universe, setUniverse] = useState('study')
  const [startDate, setStartDate] = useState('')
  const [polygonDays, setPolygonDays] = useState(5)
  const { showToast } = useToast()

  const loadStats = () => { fetch('/api/db/stats').then(r => r.json()).then(setStats).catch(() => {}) }
  useEffect(() => { loadStats() }, [])

  const buildDB = async () => {
    setBuilding(true)
    try {
      await fetch('/api/db/build', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ universe, start_date: startDate || undefined }) })
      const poll = setInterval(async () => {
        // Poll via state since db_build_progress comes via WS
        setProgress(prev => ({ ...prev, msg: 'Building...' }))
      }, 3000)
      // Simple timeout approach — backend WS would be better
      setTimeout(() => { clearInterval(poll); setBuilding(false); loadStats(); showToast('DB build complete', 'success') }, 30000)
    } catch { setBuilding(false); showToast('Build failed', 'error') }
  }

  const updateDB = async () => {
    setBuilding(true)
    try {
      await fetch('/api/db/update', { method: 'POST' })
      showToast('Daily update started', 'success')
      setTimeout(() => { setBuilding(false); loadStats() }, 15000)
    } catch { setBuilding(false); showToast('Update failed', 'error') }
  }

  const polygonUpdate = async () => {
    setBuilding(true)
    try {
      await fetch('/api/db/polygon-update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ days: polygonDays }) })
      showToast(`Polygon update (${polygonDays} days) started`, 'success')
      setTimeout(() => { setBuilding(false); loadStats() }, 20000)
    } catch { setBuilding(false); showToast('Polygon update failed', 'error') }
  }

  return (
    <div className={s.page}>
      {/* Stats */}
      <div className={s.statsBar}>
        {stats ? (
          <>
            <div className={s.stat}><span className={s.label}>Symbols</span><span className={s.value}>{Number(stats.symbol_count ?? 0).toLocaleString()}</span></div>
            <div className={s.stat}><span className={s.label}>Daily Bars</span><span className={s.value}>{Number(stats.total_rows ?? 0).toLocaleString()}</span></div>
            <div className={s.stat}><span className={s.label}>Date Range</span><span className={s.value}>{String(stats.min_date ?? '—')} → {String(stats.max_date ?? '')}</span></div>
            <div className={s.stat}><span className={s.label}>DB Size</span><span className={s.value}>{stats.size_mb ? `${(Number(stats.size_mb) / 1024).toFixed(1)} GB` : '—'}</span></div>
          </>
        ) : <div className={s.stat}><span className={s.label}>Loading...</span></div>}
      </div>

      {/* Controls */}
      <div className={s.controls}>
        <div className={s.field}><label>Universe</label>
          <select value={universe} onChange={e => setUniverse(e.target.value)}>
            <option value="study">Study (167)</option><option value="alpaca">All Tradeable</option>
          </select></div>
        <div className={s.field}><label>Start Date (optional)</label>
          <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} /></div>
        <button className={s.btnPrimary} onClick={buildDB} disabled={building}>Build DB</button>
        <button className={s.btn} onClick={updateDB} disabled={building}>Update (Alpaca)</button>
        <div className={s.polygonGroup}>
          <input type="number" min={1} max={30} value={polygonDays} onChange={e => setPolygonDays(+e.target.value)} className={s.daysInput} />
          <button className={s.btnPurple} onClick={polygonUpdate} disabled={building}>Polygon Update</button>
        </div>
      </div>

      {building && (
        <div className={s.progressBar}>
          <div className={s.progressFill} />
          <span className={s.progressText}>{progress.msg || 'Processing...'}</span>
        </div>
      )}

      <div className={s.info}>
        <p><strong>PostgreSQL</strong> — {stats ? `${Number(stats.total_rows ?? 0).toLocaleString()} daily bars across ${Number(stats.symbol_count ?? 0).toLocaleString()} symbols` : 'Loading...'}</p>
        <p>Data sources: Yahoo Finance (2006–2015), Alpaca SIP (2016+), Polygon.io (bulk updates)</p>
      </div>
    </div>
  )
}
