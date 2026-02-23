import { Clock, Wifi, WifiOff, LogOut } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useApp } from '@/context/AppContext'
import { useAuth } from '@/context/AuthContext'

function LiveClock() {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const t = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(t)
  }, [])
  return (
    <span className="font-mono text-sm text-[#64748b] flex items-center gap-1.5">
      <Clock className="w-3.5 h-3.5" />
      {now.toLocaleTimeString('en-US', { hour12: false })} ET
    </span>
  )
}

function UserBadge() {
  const { user, authEnabled, logout } = useAuth()
  if (!authEnabled || !user) return null

  const initial = (user.name || user.email || '?')[0].toUpperCase()

  return (
    <div className="flex items-center gap-2">
      {user.picture ? (
        <img
          src={user.picture}
          alt=""
          className="w-6 h-6 rounded-full border border-white/20"
          referrerPolicy="no-referrer"
        />
      ) : (
        <div className="w-6 h-6 rounded-full bg-[#00D4FF]/20 border border-[#00D4FF]/30 flex items-center justify-center text-xs font-semibold text-[#00D4FF]">
          {initial}
        </div>
      )}
      <span className="text-xs text-[#94a3b8] max-w-[120px] truncate hidden sm:inline">
        {user.name || user.email}
      </span>
      <button
        onClick={logout}
        className="p-1 rounded hover:bg-white/[0.08] text-[#64748b] hover:text-[#FF3B5C] transition-colors"
        title="Sign out"
      >
        <LogOut className="w-3.5 h-3.5" />
      </button>
    </div>
  )
}

export function TopBar() {
  const { connected } = useApp()

  return (
    <header className="h-12 flex items-center justify-between px-6 border-b border-white/[0.08] bg-[#0A0E17] shrink-0">
      <div className="flex items-center gap-4">
        <h1 className="text-base font-semibold text-white">Trading Arena</h1>
        <span className="px-2 py-0.5 rounded text-xs font-semibold bg-[#00D4FF]/20 text-[#00D4FF] border border-[#00D4FF]/30">
          Live
        </span>
      </div>
      <div className="flex items-center gap-4">
        <div
          className={`flex items-center gap-2 text-xs font-medium ${connected ? 'text-[#00D4FF]' : 'text-[#FF3B5C]'}`}
          title={connected ? 'Connected' : 'Disconnected'}
        >
          <span
            className={`w-2 h-2 rounded-full ${connected ? 'bg-[#00D4FF] shadow-[0_0_6px_rgba(0,212,255,0.6)]' : 'bg-[#FF3B5C]'}`}
          />
          {connected ? <Wifi className="w-4 h-4" /> : <WifiOff className="w-4 h-4" />}
          {connected ? 'Connected' : 'Offline'}
        </div>
        <LiveClock />
        <UserBadge />
      </div>
    </header>
  )
}
