import { Menu, Wifi, WifiOff, Clock, PanelRight, LogOut } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { HealthData } from '../lib/api'
import styles from './Topbar.module.css'

interface TopbarProps {
  health: HealthData | null
  connected: boolean
  onMenuClick: () => void
  onPanelToggle?: () => void
  pageTitle?: string
  pageSubtitle?: string
  user?: { name: string; email: string; avatar?: string; picture?: string } | null
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

function LiveClock() {
  const [time, setTime] = useState('')
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setTime(now.toLocaleTimeString('en-US', {
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        timeZone: 'America/New_York', hour12: false,
      }) + ' ET')
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])
  return <span className={`${styles.clock} font-mono`}><Clock size={12} />{time}</span>
}

function StatusPill({ status }: { status: string }) {
  const cls: Record<string, string> = {
    scanning: styles.pillRunning,
    trading: styles.pillRunning,
    waiting: styles.pillPaused,
    paused: styles.pillPaused,
    stopped: styles.pillStopped,
    standdown: styles.pillPaused,
  }
  return (
    <span className={`${styles.pill} ${cls[status] ?? styles.pillStopped}`}>
      <span className={styles.pillDot} />
      {status.toUpperCase()}
    </span>
  )
}

export default function Topbar({ health, connected, onMenuClick, onPanelToggle, pageTitle, pageSubtitle, user }: TopbarProps) {
  const equity = health?.equity
  const dailyPnl = health?.daily_pnl ?? 0
  const pnlPositive = dailyPnl >= 0

  return (
    <header className={styles.topbar}>
      <div className={styles.left}>
        <button className={styles.menuBtn} onClick={onMenuClick}>
          <Menu size={20} />
        </button>
        {pageTitle && (
          <div className={styles.titleArea}>
            <span className={styles.title}>{pageTitle}</span>
            {pageSubtitle && <span className={styles.breadcrumb}>{pageSubtitle}</span>}
          </div>
        )}
        <StatusPill status={health?.trader_status ?? 'stopped'} />
        {health?.uptime_seconds != null && (
          <span className={`${styles.uptime} font-mono`}>UP {formatUptime(health.uptime_seconds)}</span>
        )}
      </div>

      <div className={styles.right}>
        <LiveClock />
        {equity != null && (
          <span className={`${styles.equity} text-metric`}>
            ${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        )}
        <span className={`${styles.dailyPnl} font-mono ${pnlPositive ? 'text-profit' : 'text-loss'}`}>
          {pnlPositive ? '+' : ''}{dailyPnl.toFixed(2)}
        </span>
        <span className={`${styles.conn} ${connected ? styles.connOnline : styles.connOffline}`}>
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          {connected ? 'LIVE' : 'OFFLINE'}
        </span>
        {onPanelToggle && (
          <button className={styles.panelBtn} onClick={onPanelToggle} title="Toggle panel">
            <PanelRight size={16} />
          </button>
        )}
        {user && (
          <div className={styles.userArea}>
            {(user.avatar || user.picture) ? (
              <img src={user.avatar || user.picture} alt="" className={styles.userAvatar} referrerPolicy="no-referrer" />
            ) : (
              <span className={styles.userInitial}>{user.name?.charAt(0) || '?'}</span>
            )}
            <span className={styles.userName}>{user.name}</span>
            <a href="/api/auth/logout" className={styles.logoutBtn} title="Sign out">
              <LogOut size={14} />
            </a>
          </div>
        )}
      </div>
    </header>
  )
}
