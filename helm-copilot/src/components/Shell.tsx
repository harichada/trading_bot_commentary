import { NavLink, Outlet } from 'react-router-dom'
import { clsx } from 'clsx'
import { Activity, Radio, Inbox, Settings, Compass } from 'lucide-react'
import { useConnectionStatus, useStatus, useMarketIndices } from '../hooks'
import { ConnectionBanner } from './ConnectionBanner'
import { TrustStrip } from './TrustStrip'
import { IndicesStrip } from './IndicesStrip'

const navItems = [
  { to: '/', icon: Radio, label: 'Pulse' },
  { to: '/radar', icon: Compass, label: 'Radar' },
  { to: '/inbox', icon: Inbox, label: 'Inbox' },
  { to: '/settings', icon: Settings, label: 'Settings' },
]

export function Shell() {
  const { state: connState, isDemo, lastError, refresh } = useConnectionStatus()
  const { data: status, asOf: statusAsOf } = useStatus({ pollInterval: 5000 })
  const { data: indices, isDemo: indicesDemo } = useMarketIndices()

  const isLive = connState === 'connected' && !isDemo
  const isStale = indices?.stale ?? true

  return (
    <div className="min-h-screen flex flex-col bg-helm-bg">
      {/* Connection Banner */}
      <ConnectionBanner
        state={connState}
        isDemo={isDemo}
        lastError={lastError}
        onRetry={refresh}
      />

      {/* Header */}
      <header className="flex items-center justify-between px-4 py-3 border-b border-helm-border bg-helm-surface">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Activity className="w-6 h-6 text-cyan-400" />
            <h1 className="text-lg font-semibold text-white">
              Helm
            </h1>
          </div>
          <span className="text-xs text-zinc-500 hidden sm:inline">
            Personal Market Co-Pilot
          </span>
        </div>

        {/* Desktop Nav */}
        <nav className="hidden md:flex items-center gap-1">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium',
                  'transition-colors duration-150',
                  isActive
                    ? 'bg-cyan-500/20 text-cyan-400'
                    : 'text-zinc-400 hover:text-white hover:bg-helm-elevated'
                )
              }
            >
              <item.icon className="w-4 h-4" />
              {item.label}
            </NavLink>
          ))}
        </nav>
      </header>

      {/* Indices Strip */}
      <IndicesStrip
        indices={indices?.indices ?? []}
        isStale={isStale || indicesDemo}
      />

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet context={{ isDemo, isLive, connState }} />
      </main>

      {/* Mobile Nav */}
      <nav className="md:hidden flex items-center justify-around px-2 py-2 border-t border-helm-border bg-helm-surface">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              clsx(
                'flex flex-col items-center gap-1 px-3 py-2 rounded-md text-xs',
                'transition-colors duration-150',
                isActive
                  ? 'text-cyan-400'
                  : 'text-zinc-500 hover:text-white'
              )
            }
          >
            <item.icon className="w-5 h-5" />
            {item.label}
          </NavLink>
        ))}
      </nav>

      {/* Trust Strip */}
      <TrustStrip
        isLive={isLive}
        isDemo={isDemo}
        isStale={isStale}
        asOf={statusAsOf}
        botRunning={status?.bot_running ?? false}
        mode={status?.mode ?? 'not_started'}
      />
    </div>
  )
}
