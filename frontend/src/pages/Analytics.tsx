import { useState, useEffect, useRef, useCallback } from 'react'
import MetricCard from '../components/MetricCard'
import Chart from '../components/Chart'
import StatusBanner from '../components/StatusBanner'
import { usePolling } from '../hooks/usePolling'
import { useWebSocket } from '../hooks/useWebSocket'
import { useFormatters } from '../hooks/useFormatters'
import { gfApi, type GfHealth, type GfState } from '../api/client'

interface LogEntry {
  time: string
  level: string
  msg: string
}

export default function Analytics() {
  const { formatCurrency, formatPercent } = useFormatters()
  const [autoScroll, setAutoScroll] = useState(true)
  const logsEndRef = useRef<HTMLDivElement>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])

  const { data: health, error: healthError, refresh } = usePolling<GfHealth>(
    useCallback(() => gfApi.getHealth(), []),
    10000,
  )
  const { data: state } = usePolling<GfState>(
    useCallback(() => gfApi.getState(), []),
    10000,
  )

  const { lastMessage, connected: wsConnected } = useWebSocket('/gf-ws')

  // Accumulate logs from WS messages
  useEffect(() => {
    if (!lastMessage) return
    const now = new Date().toLocaleTimeString('en-US', { hour12: false, fractionalSecondDigits: 3 } as Intl.DateTimeFormatOptions)
    const msgType = String(lastMessage.type ?? 'unknown')
    const msgText = lastMessage.message as string ?? lastMessage.text as string ?? JSON.stringify(lastMessage)

    const level = msgType === 'trade' ? 'INFO'
      : msgType === 'error' ? 'ALERT'
      : msgType === 'positions_update' ? 'DEBUG'
      : msgType === 'message' ? (lastMessage.level as string ?? 'INFO')
      : 'DEBUG'

    setLogs(prev => [...prev, { time: now, level, msg: `[${msgType}] ${msgText}` }].slice(-200))
  }, [lastMessage])

  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs, autoScroll])

  // Compute metrics from live data
  const winCount = state?.win_count ?? 0
  const lossCount = state?.loss_count ?? 0
  const totalTrades = winCount + lossCount
  const winRate = totalTrades > 0 ? (winCount / totalTrades) * 100 : null
  const dailyPnl = health?.daily_pnl ?? state?.daily_pnl ?? null
  const equity = health?.equity ?? state?.equity ?? null
  const startingEquity = state?.starting_equity ?? null
  const maxDd = (startingEquity != null && equity != null && startingEquity > 0)
    ? Math.min(0, ((equity - startingEquity) / startingEquity) * 100)
    : null

  // Positions by symbol for top performers
  const positions = (state?.positions && Array.isArray(state.positions)) ? state.positions : []
  const topPerformers = positions
    .map(p => ({
      ticker: p.symbol,
      pct: p.pnl_pct ?? (p.entry_price > 0
        ? (((p.current_price ?? p.entry_price) - p.entry_price) / p.entry_price * (p.direction === 'short' ? -100 : 100))
        : 0),
      positive: (p.unrealized_pnl ?? 0) >= 0,
    }))
    .sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct))
    .slice(0, 5)

  return (
    <div className="space-y-4">
      <StatusBanner label="Gap Fade Engine" error={healthError} wsConnected={wsConnected} onRetry={refresh} />

      {/* Top Metrics */}
      <div className="grid grid-cols-5 gap-4">
        <MetricCard
          label="Win Rate"
          value={winRate != null ? `${winRate.toFixed(1)}%` : '\u2014'}
          subValue={totalTrades > 0 ? `${totalTrades} trades` : undefined}
          trend={winRate != null ? (winRate >= 50 ? 'up' : 'down') : 'neutral'}
          accentColor={winRate != null && winRate >= 50 ? '#22c55e' : '#ef4444'}
        />
        <MetricCard
          label="Win/Loss"
          value={`${winCount}W / ${lossCount}L`}
          trend="neutral"
          accentColor="#06b6d4"
        />
        <MetricCard
          label="Max Drawdown"
          value={maxDd != null ? formatPercent(maxDd) : '\u2014'}
          trend={maxDd != null ? 'down' : 'neutral'}
          accentColor="#ef4444"
        />
        <MetricCard
          label="Open Positions"
          value={String(positions.length)}
          trend="neutral"
          accentColor="#a855f7"
        />
        <MetricCard
          label="Daily P&L"
          value={dailyPnl != null ? formatCurrency(dailyPnl) : '\u2014'}
          subValue="Realized (Session)"
          trend={dailyPnl != null ? (dailyPnl >= 0 ? 'up' : 'down') : 'neutral'}
          accentColor={dailyPnl != null && dailyPnl >= 0 ? '#22c55e' : '#ef4444'}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-3 gap-4">
        {/* Equity Curve */}
        <div className="col-span-2 bg-bg-card rounded-lg border border-border p-4">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
                <span className="text-green">&#9670;</span>
                Portfolio Equity
              </h3>
              <p className="text-[10px] text-text-muted">
                Equity: {equity != null ? formatCurrency(equity) : '\u2014'}
                {startingEquity != null && ` | Starting: ${formatCurrency(startingEquity)}`}
              </p>
            </div>
          </div>
          <Chart type="area" height={280} showVolume={false} />
        </div>

        {/* Top Performers */}
        <div className="col-span-1 space-y-4">
          <div className="bg-bg-card rounded-lg border border-border p-4">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Position Performance</h3>
            {topPerformers.length === 0 ? (
              <p className="text-xs text-text-muted">No open positions</p>
            ) : (
              <div className="space-y-2">
                {topPerformers.map((p) => (
                  <div key={p.ticker} className="flex items-center gap-3">
                    <span className="text-xs font-semibold text-text-primary w-10">{p.ticker}</span>
                    <div className="flex-1 h-3 bg-bg-primary rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${p.positive ? 'bg-green' : 'bg-red'}`}
                        style={{ width: `${Math.min(Math.abs(p.pct) * 8, 100)}%` }}
                      />
                    </div>
                    <span className={`font-mono text-[10px] w-12 text-right ${p.positive ? 'text-green' : 'text-red'}`}>
                      {formatPercent(p.pct)}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Trade Stats */}
          <div className="bg-bg-card rounded-lg border border-border p-4">
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Session Stats</h3>
            <div className="space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Wins</span>
                <span className="font-mono text-green">{winCount}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Losses</span>
                <span className="font-mono text-red">{lossCount}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Total Trades</span>
                <span className="font-mono text-text-primary">{totalTrades}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-text-muted">Trading Halted</span>
                <span className={`font-mono ${health?.trading_halted ? 'text-red' : 'text-green'}`}>
                  {health?.trading_halted ? 'YES' : 'NO'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Live Logs */}
      <div className="bg-bg-card rounded-lg border border-border">
        <div className="flex items-center justify-between px-4 py-3 border-b border-border">
          <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
            <span className={`${wsConnected ? 'text-green' : 'text-red'}`}>&#9670;</span>
            Live Engine Logs
          </h3>
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green animate-pulse' : 'bg-red'}`} />
            <span className="text-[10px] text-text-muted">
              WS {wsConnected ? 'CONNECTED' : 'DISCONNECTED'} | {logs.length} entries
            </span>
            <button
              onClick={() => setAutoScroll(!autoScroll)}
              className="text-[10px] text-text-muted hover:text-text-secondary"
            >
              {autoScroll ? 'PAUSE' : 'RESUME'}
            </button>
            <button
              onClick={() => setLogs([])}
              className="text-[10px] text-text-muted hover:text-text-secondary"
            >
              CLEAR
            </button>
          </div>
        </div>
        <div className="h-64 overflow-y-auto p-4 font-mono text-xs">
          {logs.length === 0 ? (
            <p className="text-text-muted">Waiting for WebSocket messages...</p>
          ) : (
            logs.map((log, i) => (
              <div key={i} className="flex gap-3 py-0.5">
                <span className="text-text-muted shrink-0">[{log.time}]</span>
                <span className={`shrink-0 font-bold ${
                  log.level === 'INFO' ? 'text-green' :
                  log.level === 'ALERT' || log.level === 'ERROR' ? 'text-red' :
                  log.level === 'WARNING' ? 'text-yellow' :
                  log.level === 'DEBUG' ? 'text-cyan' :
                  'text-text-muted'
                }`}>
                  [{log.level}]
                </span>
                <span className="text-text-secondary">{log.msg}</span>
              </div>
            ))
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* Bottom Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-bg-card rounded-lg border border-border text-[10px]">
        <div className="flex items-center gap-2">
          <span className="text-text-muted">ENGINE STATUS</span>
          <div className="flex gap-0.5">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className={`w-1 rounded-full ${wsConnected ? 'bg-green' : 'bg-red'}`} style={{ height: `${8 + i * 3}px` }} />
            ))}
          </div>
          <span className={`font-mono ${wsConnected ? 'text-green' : 'text-red'}`}>
            {wsConnected ? 'LIVE' : 'OFFLINE'}
          </span>
        </div>
        <div className="flex items-center gap-6">
          <span className="text-text-muted">POSITIONS: <span className="text-text-primary font-mono">{positions.length}</span></span>
          <span className="text-text-muted">TRADES: <span className="text-text-primary font-mono">{totalTrades}</span></span>
          <span className="text-text-muted">WS MESSAGES: <span className="text-text-primary font-mono">{logs.length}</span></span>
        </div>
      </div>
    </div>
  )
}
