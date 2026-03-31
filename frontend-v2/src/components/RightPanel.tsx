import { useState, useEffect, useRef } from 'react'
import { useStore } from '../hooks/useStore'
import { useToast } from './Toast'
import s from './RightPanel.module.css'

type Tab = 'positions' | 'feed' | 'journal' | 'history'

function PositionsTab() {
  const store = useStore()
  const { showToast } = useToast()
  const positions = store.positions

  const closePosition = async (sym: string) => {
    if (!confirm(`Close position: ${sym}?`)) return
    try {
      await fetch(`/api/positions/${sym}/close`, { method: 'POST' })
      showToast(`Closing ${sym}`, 'success')
    } catch { showToast('Close failed', 'error') }
  }

  const adjustStop = async (sym: string, currentStop: number) => {
    const val = prompt(`New stop price for ${sym} (current: $${currentStop.toFixed(2)}):`)
    if (!val || isNaN(Number(val))) return
    try {
      await fetch(`/api/positions/${sym}/stop`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ stop_price: Number(val) }) })
      showToast(`Stop updated: ${sym} → $${val}`, 'success')
    } catch { showToast('Failed to update stop', 'error') }
  }

  if (positions.length === 0) return <div className={s.empty}>No active positions</div>

  return (
    <div className={s.tabContent}>
      {positions.map((p, i) => {
        const pnl = Number(p.pnl ?? 0)
        const sym = String(p.symbol ?? '')
        return (
          <div key={i} className={s.posRow}>
            <div className={s.posTop}>
              <span className={s.posSym}>{sym}</span>
              <span className={`${s.posDir} ${String(p.direction) === 'long' ? s.up : s.down}`}>{String(p.direction ?? '').toUpperCase()}</span>
              <span className={`${s.posPnl} ${pnl >= 0 ? s.up : s.down}`}>{pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>
              <button className={s.closeBtn} onClick={() => closePosition(sym)} title="Close">✕</button>
            </div>
            <div className={s.posDetails}>
              <span>{Number(p.remaining_shares ?? p.shares ?? 0)} shares</span>
              <span>Entry: ${Number(p.entry_price ?? 0).toFixed(2)}</span>
              <span>Now: ${Number(p.current_price ?? 0).toFixed(2)}</span>
              <span className={s.stopLink} onClick={() => adjustStop(sym, Number(p.stop_price ?? 0))}>
                Stop: ${Number(p.stop_price ?? 0).toFixed(2)}
              </span>
            </div>
          </div>
        )
      })}
    </div>
  )
}

function FeedTab() {
  const store = useStore()
  const [search, setSearch] = useState('')
  const [level, setLevel] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const feedRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll && feedRef.current) feedRef.current.scrollTop = 0
  }, [store.feed.length, autoScroll])

  const filtered = store.feed.filter(f => {
    if (search && !f.text.toLowerCase().includes(search.toLowerCase())) return false
    if (level && f.level !== level) return false
    return true
  })

  return (
    <div className={s.tabContent}>
      <div className={s.feedControls}>
        <input className={s.feedSearch} placeholder="Search..." value={search} onChange={e => setSearch(e.target.value)} />
        <select className={s.feedSelect} value={level} onChange={e => setLevel(e.target.value)}>
          <option value="">All</option>
          <option value="status">Status</option>
          <option value="scan">Scan</option>
          <option value="trade">Trade</option>
          <option value="llm">LLM</option>
          <option value="config">Config</option>
        </select>
        <label className={s.autoLabel}><input type="checkbox" checked={autoScroll} onChange={e => setAutoScroll(e.target.checked)} /> Auto</label>
      </div>
      <div className={s.feedList} ref={feedRef}>
        {filtered.map((f, i) => (
          <div key={i} className={s.feedItem}>
            <span className={s.feedTime}>{f.time}</span>
            <span className={`${s.feedLevel} ${s[`lv_${f.level}`] ?? ''}`}>{f.level.toUpperCase()}</span>
            <span className={s.feedText}>{f.text}</span>
          </div>
        ))}
        {filtered.length === 0 && <div className={s.empty}>No feed items</div>}
      </div>
    </div>
  )
}

function JournalTab() {
  const [entries, setEntries] = useState<Record<string, unknown>[]>([])
  const [filter, setFilter] = useState('')
  const [budget, setBudget] = useState<Record<string, unknown> | null>(null)

  const load = () => {
    fetch('/api/journal').then(r => r.json()).then(d => setEntries(Array.isArray(d) ? d : d.entries ?? [])).catch(() => {})
    fetch('/api/events').then(r => r.json()).then(setBudget).catch(() => {})
  }

  useEffect(load, [])

  const filtered = entries.filter(e => !filter || String(e.type ?? '') === filter)

  return (
    <div className={s.tabContent}>
      <div className={s.feedControls}>
        <select className={s.feedSelect} value={filter} onChange={e => setFilter(e.target.value)}>
          <option value="">All types</option>
          <option value="observation">Observations</option>
          <option value="reasoning">Reasoning</option>
          <option value="action">Actions</option>
          <option value="no_action">No-Actions</option>
          <option value="reflection">Reflections</option>
        </select>
        <button className={s.refreshBtn} onClick={load}>↻ Refresh</button>
        {budget && <span className={s.budgetText}>{String(budget.calls ?? 0)}/{String(budget.limit ?? '—')} calls</span>}
      </div>
      <div className={s.feedList}>
        {filtered.slice(0, 50).map((e, i) => (
          <div key={i} className={s.journalEntry}>
            <span className={s.journalType}>{String(e.type ?? '')}</span>
            <span className={s.journalText}>{String(e.text ?? e.message ?? e.content ?? '')}</span>
          </div>
        ))}
        {filtered.length === 0 && <div className={s.empty}>No journal entries</div>}
      </div>
    </div>
  )
}

function HistoryTab() {
  const [history, setHistory] = useState<Record<string, unknown>[]>([])
  const { showToast } = useToast()

  const load = () => {
    fetch('/api/config/history').then(r => r.json()).then(d => setHistory(Array.isArray(d) ? d : d.history ?? [])).catch(() => {})
  }

  useEffect(load, [])

  const rollback = async (id: string) => {
    try {
      await fetch(`/api/config/rollback/${id}`, { method: 'POST' })
      showToast('Config rolled back', 'success')
      load()
    } catch { showToast('Rollback failed', 'error') }
  }

  return (
    <div className={s.tabContent}>
      <button className={s.refreshBtn} onClick={load} style={{ marginBottom: 8 }}>↻ Refresh</button>
      <div className={s.feedList}>
        {history.slice(0, 30).map((h, i) => (
          <div key={i} className={s.historyEntry}>
            <div className={s.historyTop}>
              <span className={s.historyTime}>{String(h.timestamp ?? h.time ?? '')}</span>
              <span className={s.historySource}>{String(h.source ?? '')}</span>
              {h.id && <button className={s.rollbackBtn} onClick={() => rollback(String(h.id))}>Rollback</button>}
            </div>
            <div className={s.historyChanges}>
              {Array.isArray(h.changes) ? (h.changes as Record<string, unknown>[]).map((c, j) => (
                <div key={j} className={s.historyChange}>{String(c.field ?? c.key ?? '')}: {String(c.old ?? '')} → {String(c.new ?? c.value ?? '')}</div>
              )) : <span className={s.historyChange}>{String(h.description ?? h.message ?? JSON.stringify(h.changes ?? ''))}</span>}
            </div>
          </div>
        ))}
        {history.length === 0 && <div className={s.empty}>No config history</div>}
      </div>
    </div>
  )
}

export default function RightPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [tab, setTab] = useState<Tab>('positions')
  const store = useStore()

  if (!open) return null

  return (
    <div className={s.panel}>
      <div className={s.header}>
        <div className={s.tabs}>
          {(['positions', 'feed', 'journal', 'history'] as Tab[]).map(t => (
            <button key={t} className={`${s.tab} ${tab === t ? s.tabActive : ''}`} onClick={() => setTab(t)}>
              {t === 'positions' ? `Pos (${store.positions.length})` : t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>
        <button className={s.closePanelBtn} onClick={onClose}>✕</button>
      </div>
      {tab === 'positions' && <PositionsTab />}
      {tab === 'feed' && <FeedTab />}
      {tab === 'journal' && <JournalTab />}
      {tab === 'history' && <HistoryTab />}
    </div>
  )
}
