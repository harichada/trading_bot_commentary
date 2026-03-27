import { useState, useEffect, type ReactNode } from 'react'
import { NavLink, useNavigate, useLocation } from 'react-router-dom'
import { gfApi, type GfHealth } from '../api/client'

interface LayoutProps {
  children: ReactNode
}

const navItems = [
  { path: '/', icon: 'grid', label: 'Dashboard' },
  { path: '/markets', icon: 'chart', label: 'Markets' },
  { path: '/trade', icon: 'trade', label: 'Trade' },
  { path: '/positions', icon: 'positions', label: 'Positions' },
  { path: '/orders', icon: 'orders', label: 'Orders' },
  { path: '/strategies', icon: 'strategy', label: 'Strategies' },
  { path: '/analytics', icon: 'analytics', label: 'Analytics' },
  { path: '/logs', icon: 'logs', label: 'Logs' },
  { path: '/config', icon: 'config', label: 'Config' },
  { path: '/tracker', icon: 'tracker', label: 'Tracker' },
  { path: '/ai', icon: 'ai', label: 'AI Chat' },
]

function NavIcon({ icon, className = '' }: { icon: string; className?: string }) {
  const icons: Record<string, ReactNode> = {
    grid: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M3 3h6v6H3V3zm8 0h6v6h-6V3zM3 11h6v6H3v-6zm8 0h6v6h-6v-6z" />
      </svg>
    ),
    chart: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M2 16l5-5 3 3 8-8v4h2V2h-8v2h4l-6 6-3-3-7 7 2 2z" />
      </svg>
    ),
    trade: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M4 4h3v12H4V4zm5 4h3v8H9V8zm5-2h3v10h-3V6z" />
      </svg>
    ),
    positions: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M3 3h14v2H3V3zm0 4h14v2H3V7zm0 4h14v2H3v-2zm0 4h14v2H3v-2z" />
      </svg>
    ),
    orders: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M4 4h12v2H4V4zm0 4h12v2H4V8zm0 4h8v2H4v-2zm10 0l3 3-3 3v-6z" />
      </svg>
    ),
    strategy: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M10 2L2 7l8 5 8-5-8-5zM2 13l8 5 8-5-2-1.3L10 15l-6-3.7L2 13z" />
      </svg>
    ),
    analytics: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm0 2a6 6 0 110 12V4z" />
      </svg>
    ),
    logs: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M3 4h14v1H3V4zm0 3h10v1H3V7zm0 3h14v1H3v-1zm0 3h8v1H3v-1zm0 3h14v1H3v-1z" />
      </svg>
    ),
    config: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M10 13a3 3 0 100-6 3 3 0 000 6zm7.5-2.5h-1.8a5.5 5.5 0 00-.5-1.2l1.3-1.3-1.5-1.5-1.3 1.3a5.5 5.5 0 00-1.2-.5V5.5h-2v1.8a5.5 5.5 0 00-1.2.5L7 6.5 5.5 8l1.3 1.3a5.5 5.5 0 00-.5 1.2H4.5v2h1.8c.1.4.3.8.5 1.2L5.5 15 7 16.5l1.3-1.3c.4.2.8.4 1.2.5v1.8h2v-1.8c.4-.1.8-.3 1.2-.5l1.3 1.3 1.5-1.5-1.3-1.3c.2-.4.4-.8.5-1.2h1.8v-2z" />
      </svg>
    ),
    tracker: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm1 12H9v-2h2v2zm0-4H9V6h2v4z" />
      </svg>
    ),
    ai: (
      <svg className={className} viewBox="0 0 20 20" fill="currentColor">
        <path d="M10 2a6 6 0 00-6 6c0 2.2 1.2 4.1 3 5.2V16a1 1 0 001 1h4a1 1 0 001-1v-2.8c1.8-1.1 3-3 3-5.2a6 6 0 00-6-6zm-1 15v1h2v-1H9z" />
      </svg>
    ),
  }
  return <>{icons[icon] ?? icons.grid}</>
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  if (h > 0) return `${h}h ${m}m`
  return `${m}m`
}

export default function Layout({ children }: LayoutProps) {
  const [health, setHealth] = useState<GfHealth | null>(null)
  const [gfOnline, setGfOnline] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const h = await gfApi.getHealth()
        setHealth(h)
        setGfOnline(true)
      } catch {
        setGfOnline(false)
      }
    }
    fetchHealth()
    const timer = setInterval(fetchHealth, 10000)
    return () => clearInterval(timer)
  }, [])

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const location = useLocation()

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false)
  }, [location.pathname])

  const marketOpen = health?.trading_halted === false
  const equity = typeof health?.equity === 'number' ? health.equity : null

  return (
    <div className="h-full flex bg-bg-primary">
      {/* Mobile overlay */}
      {mobileMenuOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-40 md:hidden"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* Sidebar — hidden on mobile, shown on md+ */}
      <aside className={`
        ${mobileMenuOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0 fixed md:static z-50
        w-56 md:w-16 h-full
        flex flex-col items-center py-4
        bg-bg-card border-r border-border shrink-0
        transition-transform duration-200 ease-in-out
      `}>
        {/* Logo */}
        <button
          onClick={() => navigate('/')}
          className="w-10 h-10 rounded-xl bg-green/10 flex items-center justify-center mb-6"
        >
          <svg viewBox="0 0 24 24" className="w-6 h-6 text-green" fill="currentColor">
            <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.3L19.5 8 12 11.7 4.5 8 12 4.3zM4 9.3l7 3.5V19l-7-3.5V9.3zm9 9.7v-6.2l7-3.5V16L13 19z" />
          </svg>
        </button>

        {/* Nav Items */}
        <nav className="flex-1 flex flex-col gap-1 w-full px-2">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `group relative flex items-center md:justify-center w-full md:w-12 h-10 rounded-lg transition-colors md:mx-auto px-3 md:px-0 ${
                  isActive
                    ? 'bg-green/10 text-green'
                    : 'text-text-muted hover:text-text-secondary hover:bg-bg-card-hover'
                }`
              }
              title={item.label}
            >
              <NavIcon icon={item.icon} className="w-5 h-5" />
              {/* Label — visible on mobile sidebar, tooltip on desktop */}
              <span className="md:hidden ml-3 text-sm">{item.label}</span>
              <span className="hidden md:block absolute left-full ml-2 px-2 py-1 text-xs bg-bg-card border border-border rounded opacity-0 group-hover:opacity-100 pointer-events-none whitespace-nowrap z-50 text-text-primary transition-opacity">
                {item.label}
              </span>
            </NavLink>
          ))}
        </nav>

        {/* Bottom nav */}
        <div className="flex flex-col gap-1 w-full px-2">
          <button className="flex items-center justify-center w-12 h-10 rounded-lg text-text-muted hover:text-text-secondary hover:bg-bg-card-hover transition-colors mx-auto" title="Support">
            <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M10 2a8 8 0 100 16 8 8 0 000-16zm1 12H9v-2h2v2zm0-4H9V6h2v4z" />
            </svg>
          </button>
          <button className="flex items-center justify-center w-12 h-10 rounded-lg text-text-muted hover:text-text-secondary hover:bg-bg-card-hover transition-colors mx-auto" title="Settings">
            <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path d="M10 13a3 3 0 100-6 3 3 0 000 6z" />
            </svg>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="h-12 flex items-center justify-between px-4 bg-bg-card border-b border-border shrink-0">
          <div className="flex items-center gap-3">
            {/* Hamburger — mobile only */}
            <button
              className="md:hidden p-1.5 rounded-lg text-text-muted hover:text-text-primary hover:bg-bg-card-hover"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                {mobileMenuOpen ? (
                  <path d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" />
                ) : (
                  <path d="M3 5h14M3 10h14M3 15h14" stroke="currentColor" strokeWidth="2" fill="none" />
                )}
              </svg>
            </button>
            <h1 className="text-sm font-semibold text-text-primary">
              <span className="text-green">Rudra</span> <span className="hidden sm:inline">Engine</span>
            </h1>
            <span className="hidden lg:inline text-[10px] text-text-muted italic">INSTITUTIONAL TERMINAL</span>
            {gfOnline ? (
              <span className={`flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded ${
                marketOpen ? 'bg-green/10 text-green' : 'bg-yellow/10 text-yellow'
              }`}>
                <span className={`w-1.5 h-1.5 rounded-full ${marketOpen ? 'bg-green animate-pulse' : 'bg-yellow'}`} />
                {health?.trader_status?.toUpperCase() ?? (marketOpen ? 'ACTIVE' : 'HALTED')}
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-[10px] font-semibold px-2 py-0.5 rounded bg-red/10 text-red">
                <span className="w-1.5 h-1.5 rounded-full bg-red" />
                OFFLINE
              </span>
            )}
            {health?.uptime_seconds != null && (
              <span className="text-[10px] text-text-muted">
                UP: {formatUptime(health.uptime_seconds as number)}
              </span>
            )}
          </div>

          <div className="flex items-center gap-4">
            <span className="font-mono text-sm text-text-primary font-semibold">
              {equity != null ? `$${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '\u2014'}
            </span>

            <span className="text-[10px] text-text-muted flex items-center gap-1">
              <svg viewBox="0 0 12 12" className={`w-3 h-3 ${gfOnline ? 'text-green' : 'text-red'}`} fill="currentColor">
                <circle cx="6" cy="6" r="3" />
              </svg>
              {gfOnline ? 'CONNECTED' : 'DISCONNECTED'}
            </span>

            <div className="flex gap-1">
              <button className="px-3 py-1 text-[10px] font-bold rounded bg-green text-black hover:bg-green/90 transition-colors">
                BUY / LONG
              </button>
              <button className="px-3 py-1 text-[10px] font-bold rounded bg-red text-white hover:bg-red/90 transition-colors">
                SELL / SHORT
              </button>
            </div>

            {/* Notification bell */}
            <button className="relative text-text-muted hover:text-text-secondary transition-colors">
              <svg viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                <path d="M10 2a6 6 0 00-6 6v3l-2 2v1h16v-1l-2-2V8a6 6 0 00-6-6zm-1 16h2a1 1 0 01-2 0z" />
              </svg>
              <span className="absolute -top-0.5 -right-0.5 w-2 h-2 bg-red rounded-full" />
            </button>

            {/* Avatar */}
            <div className="w-7 h-7 rounded-full bg-purple/20 flex items-center justify-center text-[10px] font-bold text-purple">
              R
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto p-4">
          {children}
        </main>
      </div>
    </div>
  )
}
