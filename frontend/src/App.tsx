import { useState } from 'react'
import { AuthProvider, useAuth } from '@/context/AuthContext'
import { AppProvider, useApp } from '@/context/AppContext'
import { ToastProvider } from '@/context/ToastContext'
import { LoginPage } from '@/components/auth/LoginPage'
import { AccessDeniedPage } from '@/components/auth/AccessDeniedPage'
import { TopBar } from '@/components/layout/TopBar'
import { Sidebar, type NavId } from '@/components/layout/Sidebar'
import { SummaryBar } from '@/components/layout/SummaryBar'
import { StrategyCardsRow } from '@/components/dashboard/StrategyCardsRow'
import { ControlStrip } from '@/components/dashboard/ControlStrip'
import { EquityChart } from '@/components/charts/EquityChart'
import { PositionsPanel } from '@/components/trading/PositionsPanel'
import { TradeHistoryTable } from '@/components/trading/TradeHistoryTable'
import { BacktestPanel } from '@/components/backtest/BacktestPanel'
import { Panel } from '@/components/ui/Panel'

function AuthGate({ children }: { children: React.ReactNode }) {
  const { user, loading, authEnabled } = useAuth()

  // Auth not configured — open access (same as before)
  if (!authEnabled) return <>{children}</>

  // Loading spinner
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0A0E17]">
        <div className="w-8 h-8 border-2 border-[#00D4FF]/30 border-t-[#00D4FF] rounded-full animate-spin" />
      </div>
    )
  }

  // Check for access-denied hash route
  if (window.location.hash === '#/access-denied') {
    return <AccessDeniedPage />
  }

  // Not logged in
  if (!user) return <LoginPage />

  return <>{children}</>
}

function MainContent() {
  const [nav, setNav] = useState<NavId>('dashboard')
  const { start } = useApp()

  const showArena = nav === 'dashboard' || nav === 'arena'

  return (
    <div className="flex flex-col h-full min-h-0 bg-[#0A0E17]">
      <TopBar />
      <div className="flex flex-1 min-h-0">
        <Sidebar active={nav} onSelect={setNav} onStartTrading={start} />
        <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
          {showArena && (
            <>
              <div className="px-6 py-3 border-b border-white/[0.08]">
                <StrategyCardsRow />
              </div>
              <div className="flex-1 flex min-h-0 p-4 gap-4">
                <div className="flex-1 flex flex-col min-w-0 rounded-lg border border-white/[0.08] bg-white/[0.02] overflow-hidden">
                  <div className="px-4 py-2 border-b border-white/[0.08] flex items-center justify-between shrink-0">
                    <h2 className="text-xs font-semibold text-[#64748b] uppercase tracking-wider">Performance</h2>
                    <ControlStrip />
                  </div>
                  <div className="flex-1 min-h-[260px] p-2">
                    <EquityChart height={320} />
                  </div>
                </div>
                <div className="w-[380px] shrink-0 min-h-0">
                  <PositionsPanel />
                </div>
              </div>
            </>
          )}
          {nav === 'scanner' && <ScannerView />}
          {nav === 'trades' && <TradeHistoryView />}
          {nav === 'backtest' && <BacktestView />}
          {nav === 'config' && <ConfigView />}
          {nav === 'guide' && <GuideView />}
        </div>
      </div>
      <SummaryBar />
    </div>
  )
}

function ScannerView() {
  const { state } = useApp()
  const candidates = state?.candidates ?? []

  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="rounded-lg border border-white/[0.08] bg-white/[0.02] overflow-hidden">
        <div className="px-4 py-3 border-b border-white/[0.08] flex items-center justify-between">
          <h2 className="text-sm font-semibold text-white uppercase tracking-wider">Gap Candidates</h2>
          <span className="text-xs text-[#64748b]">Last: {state?.last_scan_time ?? '—'}</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-white/[0.08] text-[#64748b] text-left text-[10px] uppercase tracking-wider">
                <th className="pb-2 pr-3 font-medium">Symbol</th>
                <th className="pb-2 pr-3 font-medium">Dir</th>
                <th className="pb-2 pr-3 font-medium">Gap %</th>
                <th className="pb-2 pr-3 font-medium">Prev Close</th>
                <th className="pb-2 pr-3 font-medium">Current</th>
                <th className="pb-2 pr-3 font-medium">Vol Ratio</th>
                <th className="pb-2 font-medium">Score</th>
              </tr>
            </thead>
            <tbody>
              {candidates.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-[#64748b]">Run a scan to see candidates</td>
                </tr>
              ) : (
                candidates.slice(0, 20).map((c: any) => (
                  <tr key={c.symbol} className="border-b border-white/[0.04] hover:bg-white/[0.04]">
                    <td className="py-2.5 pr-3 font-semibold text-white">{c.symbol}</td>
                    <td className={`py-2.5 pr-3 font-semibold ${c.direction === 'short' ? 'text-[#FF3B5C]' : 'text-[#00D4FF]'}`}>
                      {(c.direction ?? '').toUpperCase()}
                    </td>
                    <td className="py-2.5 pr-3 font-mono">{(Number(c.gap_pct) * 100).toFixed(1)}%</td>
                    <td className="py-2.5 pr-3 font-mono text-[#94a3b8]">${Number(c.prev_close).toFixed(2)}</td>
                    <td className="py-2.5 pr-3 font-mono">${Number(c.current ?? c.premarket_price ?? 0).toFixed(2)}</td>
                    <td className="py-2.5 pr-3 font-mono">{Number(c.vol_ratio).toFixed(2)}</td>
                    <td className="py-2.5 font-mono font-semibold">{Number(c.score).toFixed(2)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function TradeHistoryView() {
  return (
    <div className="flex-1 overflow-auto p-6">
      <TradeHistoryTable />
    </div>
  )
}

function BacktestView() {
  return (
    <div className="flex-1 overflow-auto p-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <BacktestPanel />
        <Panel title="Results">Run a backtest to see results.</Panel>
      </div>
    </div>
  )
}

function ConfigView() {
  return (
    <div className="flex-1 overflow-auto p-6">
      <Panel title="Configuration">
        <p className="text-[#64748b] text-sm">Strategy parameters (gap %, stop %, max positions, etc.) can be edited here. API integration pending.</p>
      </Panel>
    </div>
  )
}

function GuideView() {
  return (
    <div className="flex-1 overflow-auto p-6">
      <Panel title="Guide">
        <p className="text-[#64748b] text-sm mb-2">Gap Fade: Short gap-ups on below-average volume. Study: 71% fade rate.</p>
        <ul className="text-sm text-[#64748b] space-y-1 list-disc list-inside">
          <li>Scan: Pre-market scan for gap candidates</li>
          <li>Start: Begin live trading loop</li>
          <li>Pause / Resume / Stop: Control the loop</li>
          <li>Reset: Clear equity and trade log</li>
        </ul>
      </Panel>
    </div>
  )
}

export default function App() {
  return (
    <AuthProvider>
      <AuthGate>
        <ToastProvider>
          <AppProvider>
            <MainContent />
          </AppProvider>
        </ToastProvider>
      </AuthGate>
    </AuthProvider>
  )
}
