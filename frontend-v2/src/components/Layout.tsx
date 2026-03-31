import { useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import ErrorBoundary from './ErrorBoundary'
import Sidebar from './Sidebar'
import Topbar from './Topbar'
import RightPanel from './RightPanel'
import StatusBar from './StatusBar'
import { useTraderState } from '../hooks/useTraderState'
import styles from './Layout.module.css'

const pageMeta: Record<string, { title: string; subtitle: string }> = {
  '/':             { title: 'Dashboard',     subtitle: 'Real-time gap fade signals & execution' },
  '/scanner':      { title: 'Scanner',       subtitle: 'Pre-market gap candidates' },
  '/trades':       { title: 'Trade Log',     subtitle: 'Completed trades & P&L breakdown' },
  '/intraday':     { title: 'Intraday',      subtitle: 'Live intraday strategies' },
  '/tracker':      { title: 'Tracker',       subtitle: 'Position tracking & manual orders' },
  '/backtest':     { title: 'Backtest',      subtitle: 'Historical strategy validation' },
  '/walkforward':  { title: 'Walk-Forward',  subtitle: 'Out-of-sample optimization' },
  '/intraday-wf':  { title: 'Intraday WF',  subtitle: 'Intraday walk-forward analysis' },
  '/strategies':   { title: 'Strategies',    subtitle: 'Strategy configuration & registry' },
  '/config':       { title: 'Config',        subtitle: 'Engine parameters & risk limits' },
  '/database':     { title: 'Database',      subtitle: 'Price data & schema management' },
  '/chat':         { title: 'Rudra Chat',    subtitle: 'AI trading assistant' },
  '/guide':        { title: 'Guide',         subtitle: 'Documentation & release notes' },
  '/lab':          { title: 'Strategy Lab',  subtitle: 'Intraday strategy development & replay' },
  '/replay':       { title: 'Visual Replay', subtitle: 'Watch strategies trade bar-by-bar' },
}

export default function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [rightPanelOpen, setRightPanelOpen] = useState(false)
  const { health, connected } = useTraderState(5000)
  const location = useLocation()

  const meta = pageMeta[location.pathname] ?? { title: 'Rudra', subtitle: '' }
  const botLive = health?.trader_status === 'scanning' || health?.trader_status === 'trading'

  return (
    <div className={`${styles.shell} ${sidebarCollapsed ? styles.collapsed : ''}`}>
      <Sidebar
        open={sidebarOpen}
        collapsed={sidebarCollapsed}
        onClose={() => setSidebarOpen(false)}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
        botLive={botLive}
      />
      <div className={styles.main}>
        <Topbar
          health={health}
          connected={connected}
          onMenuClick={() => setSidebarOpen(!sidebarOpen)}
          onPanelToggle={() => setRightPanelOpen(!rightPanelOpen)}
          pageTitle={meta.title}
          pageSubtitle={meta.subtitle}
        />
        <div className={styles.body}>
          <main className={styles.content} key={location.pathname}>
            <ErrorBoundary key={location.pathname}>
              <Outlet />
            </ErrorBoundary>
          </main>
          <RightPanel open={rightPanelOpen} onClose={() => setRightPanelOpen(false)} />
        </div>
        <StatusBar />
      </div>
    </div>
  )
}
