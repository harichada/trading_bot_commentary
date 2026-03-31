import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  LayoutDashboard, ScanSearch, Receipt, CandlestickChart, Activity,
  BarChart2, GitBranch, Layers, Settings2, Database, Bot, BookOpen,
  Play, Square, RefreshCw,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { api } from '../lib/api'
import styles from './CommandPalette.module.css'

interface Command {
  group: string
  label: string
  icon: LucideIcon
  action: () => void
  shortcut?: string
}

export default function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState(0)
  const inputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const nav = useCallback((path: string) => { navigate(path); setOpen(false) }, [navigate])

  const commands: Command[] = [
    // Navigate
    { group: 'Navigate', label: 'Dashboard',    icon: LayoutDashboard, action: () => nav('/'),            shortcut: '1' },
    { group: 'Navigate', label: 'Scanner',      icon: ScanSearch,      action: () => nav('/scanner'),     shortcut: '2' },
    { group: 'Navigate', label: 'Trade Log',    icon: Receipt,         action: () => nav('/trades'),      shortcut: '3' },
    { group: 'Navigate', label: 'Backtest',     icon: BarChart2,       action: () => nav('/backtest'),    shortcut: '4' },
    { group: 'Navigate', label: 'Strategies',   icon: Layers,          action: () => nav('/strategies'),  shortcut: '5' },
    { group: 'Navigate', label: 'Config',       icon: Settings2,       action: () => nav('/config'),      shortcut: '6' },
    { group: 'Navigate', label: 'Rudra Chat',   icon: Bot,             action: () => nav('/chat'),        shortcut: '7' },
    { group: 'Navigate', label: 'Intraday',     icon: CandlestickChart,action: () => nav('/intraday') },
    { group: 'Navigate', label: 'Tracker',      icon: Activity,        action: () => nav('/tracker') },
    { group: 'Navigate', label: 'Walk-Forward', icon: GitBranch,       action: () => nav('/walkforward') },
    { group: 'Navigate', label: 'Database',     icon: Database,        action: () => nav('/database') },
    { group: 'Navigate', label: 'Guide',        icon: BookOpen,        action: () => nav('/guide') },
    // Actions
    { group: 'Actions', label: 'Start Bot',       icon: Play,      action: () => { api.start(); setOpen(false) } },
    { group: 'Actions', label: 'Stop Bot',        icon: Square,    action: () => { api.stop(); setOpen(false) } },
    { group: 'Actions', label: 'Run Scan',        icon: RefreshCw, action: () => { api.scan(); setOpen(false) } },
  ]

  const filtered = query
    ? commands.filter(c => c.label.toLowerCase().includes(query.toLowerCase()))
    : commands

  // Keyboard: Ctrl+K to toggle, Escape to close, arrow keys, Enter
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault()
        setOpen(v => !v)
        setQuery('')
        setSelected(0)
      }
      if (e.key === 'Escape' && open) setOpen(false)
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [open])

  useEffect(() => {
    if (open) inputRef.current?.focus()
  }, [open])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelected(s => Math.min(s + 1, filtered.length - 1)) }
    if (e.key === 'ArrowUp') { e.preventDefault(); setSelected(s => Math.max(s - 1, 0)) }
    if (e.key === 'Enter' && filtered[selected]) { filtered[selected].action() }
  }

  if (!open) return null

  // Group commands
  const groups = new Map<string, Command[]>()
  filtered.forEach(c => {
    if (!groups.has(c.group)) groups.set(c.group, [])
    groups.get(c.group)!.push(c)
  })

  let idx = 0

  return (
    <div className={styles.overlay} onClick={(e) => { if (e.target === e.currentTarget) setOpen(false) }}>
      <div className={styles.palette}>
        <input
          ref={inputRef}
          className={styles.input}
          placeholder="Navigate pages, run actions..."
          value={query}
          onChange={e => { setQuery(e.target.value); setSelected(0) }}
          onKeyDown={onKeyDown}
        />
        <div className={styles.list}>
          {[...groups.entries()].map(([group, cmds]) => (
            <div key={group}>
              <div className={styles.groupLabel}>{group}</div>
              {cmds.map(cmd => {
                const i = idx++
                return (
                  <div
                    key={cmd.label}
                    className={`${styles.item} ${i === selected ? styles.selected : ''}`}
                    onClick={cmd.action}
                    onMouseEnter={() => setSelected(i)}
                  >
                    <cmd.icon size={16} className={styles.itemIcon} />
                    {cmd.label}
                    {cmd.shortcut && <span className={styles.shortcut}>{cmd.shortcut}</span>}
                  </div>
                )
              })}
            </div>
          ))}
          {filtered.length === 0 && (
            <div className={styles.empty}>No results</div>
          )}
        </div>
        <div className={styles.footer}>
          <span className={styles.hint}>↑↓ navigate</span>
          <span className={styles.hint}>↵ select</span>
          <span className={styles.hint}>esc close</span>
        </div>
      </div>
    </div>
  )
}
