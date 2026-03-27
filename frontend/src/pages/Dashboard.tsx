import { useState, useEffect, useCallback } from 'react'
import MetricCard from '../components/MetricCard'
import Chart from '../components/Chart'
import StatusBanner from '../components/StatusBanner'
import { useFormatters } from '../hooks/useFormatters'
import { usePolling } from '../hooks/usePolling'
import { useWebSocket } from '../hooks/useWebSocket'
import { gfApi, type GfHealth, type GfState, type GfPosition } from '../api/client'
import type { Time } from 'lightweight-charts'

interface EquityPoint {
  time: Time
  value: number
}

interface ExecLogEntry {
  side: string
  ticker: string
  price: number
  shares: number
  time: string
}

export default function Dashboard() {
  const { formatCurrency, formatPercent } = useFormatters()

  // Polling for health + state
  const { data: health, error: healthError, refresh: refreshHealth } = usePolling<GfHealth>(
    useCallback(() => gfApi.getHealth(), []),
    10000,
  )
  const { data: state, error: stateError } = usePolling<GfState>(
    useCallback(() => gfApi.getState(), []),
    10000,
  )

  // WebSocket for real-time updates
  const { lastMessage, connected: wsConnected } = useWebSocket('/gf-ws')

  // Live positions from state + WS updates
  const [positions, setPositions] = useState<GfPosition[]>([])
  const [equityCurve, setEquityCurve] = useState<EquityPoint[]>([])
  const [execLog, setExecLog] = useState<ExecLogEntry[]>([])
  const [wsMessages, setWsMessages] = useState<Array<{ type: string; text: string; time: string }>>([])

  // Update positions from state
  useEffect(() => {
    if (state?.positions && Array.isArray(state.positions)) {
      setPositions(state.positions)
    }
  }, [state])

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return
    const msg = lastMessage

    if (msg.type === 'positions_update' || msg.type === 'live_status') {
      if (msg.positions && Array.isArray(msg.positions)) {
        setPositions(msg.positions as GfPosition[])
      }
      // Build equity curve from updates
      const eq = typeof msg.equity === 'number' ? msg.equity :
                 (msg.stats && typeof (msg.stats as Record<string, unknown>).equity === 'number')
                   ? (msg.stats as Record<string, unknown>).equity as number : null
      if (eq != null) {
        const now = Math.floor(Date.now() / 1000) as Time
        setEquityCurve(prev => {
          const next = [...prev, { time: now, value: eq }]
          return next.length > 500 ? next.slice(-500) : next
        })
      }
    }

    if (msg.type === 'trade') {
      const trades = Array.isArray(msg.trades) ? msg.trades : [msg]
      for (const t of trades) {
        const trade = t as Record<string, unknown>
        setExecLog(prev => [{
          side: (trade.direction as string ?? trade.side as string ?? 'BUY').toUpperCase(),
          ticker: trade.symbol as string ?? '???',
          price: trade.exit_price as number ?? trade.price as number ?? 0,
          shares: trade.qty as number ?? 0,
          time: new Date().toLocaleTimeString(),
        }, ...prev].slice(0, 50))
      }
    }

    if (msg.type === 'message' || msg.type === 'log') {
      setWsMessages(prev => [{
        type: msg.level as string ?? 'INFO',
        text: msg.message as string ?? msg.text as string ?? JSON.stringify(msg),
        time: new Date().toLocaleTimeString(),
      }, ...prev].slice(0, 20))
    }
  }, [lastMessage])

  const error = healthError ?? stateError

  // Compute dashboard metrics from live data
  const equity = health?.equity ?? state?.equity ?? null
  const dailyPnl = health?.daily_pnl ?? state?.daily_pnl ?? null
  const startingEquity = state?.starting_equity ?? null
  const dailyPnlPct = (dailyPnl != null && startingEquity != null && startingEquity > 0)
    ? (dailyPnl / startingEquity) * 100 : null
  const posCount = positions.length
  const tradingHalted = health?.trading_halted ?? false
  const traderStatus = health?.trader_status ?? 'unknown'

  const winCount = state?.win_count ?? 0
  const lossCount = state?.loss_count ?? 0
  const totalTrades = winCount + lossCount
  const winRate = totalTrades > 0 ? (winCount / totalTrades) * 100 : null

  // Build area chart data from equity curve
  const areaChartData = equityCurve.length > 1
    ? equityCurve.map(p => ({
        time: p.time,
        open: p.value,
        high: p.value,
        low: p.value,
        close: p.value,
      }))
    : undefined

  return (
    <div className="space-y-4">
      <StatusBanner
        label="Gap Fade Engine"
        error={error}
        wsConnected={wsConnected}
        onRetry={refreshHealth}
      />

      {/* Top Stats Row */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          label="Total Equity"
          value={equity != null ? formatCurrency(equity) : '\u2014'}
          trend="neutral"
        />
        <MetricCard
          label="Session P&L"
          value={dailyPnl != null ? formatCurrency(dailyPnl) : '\u2014'}
          subValue={dailyPnlPct != null ? formatPercent(dailyPnlPct) : undefined}
          trend={dailyPnl != null ? (dailyPnl >= 0 ? 'up' : 'down') : 'neutral'}
        />
        <MetricCard
          label="Win Rate"
          value={winRate != null ? `${winRate.toFixed(1)}%` : '\u2014'}
          subValue={totalTrades > 0 ? `${totalTrades} trades` : undefined}
          trend={winRate != null ? (winRate >= 50 ? 'up' : 'down') : 'neutral'}
        />
        <MetricCard
          label="Open Positions"
          value={String(posCount)}
          subValue={tradingHalted ? 'HALTED' : traderStatus.toUpperCase()}
          trend={tradingHalted ? 'down' : 'neutral'}
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-3 gap-4">
        {/* NAV Chart */}
        <div className="col-span-2 bg-bg-card rounded-lg border border-border p-4">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wider">Intraday Equity Curve</h2>
              <p className="text-[10px] text-text-muted">
                {equityCurve.length > 0
                  ? `${equityCurve.length} data points from WebSocket`
                  : 'Waiting for WebSocket equity updates...'}
              </p>
            </div>
          </div>
          {equity != null && (
            <div className="bg-bg-card rounded-lg p-2 mb-2 inline-block">
              <p className="text-[10px] text-text-muted uppercase">Current Equity</p>
              <p className="font-mono text-lg font-semibold text-text-primary">{formatCurrency(equity)}</p>
            </div>
          )}
          <Chart type="area" height={320} showVolume={false} data={areaChartData} />
        </div>

        {/* Activity Feed from WS */}
        <div className="col-span-1">
          <div className="bg-bg-card rounded-lg border border-border p-4">
            <div className="flex items-center gap-2 mb-4">
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green animate-pulse' : 'bg-red'}`} />
              <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider">Live Activity</h3>
            </div>

            {wsMessages.length === 0 && (
              <p className="text-xs text-text-muted">Waiting for WebSocket messages...</p>
            )}

            <div className="space-y-2 max-h-80 overflow-y-auto">
              {wsMessages.map((msg, i) => (
                <div key={i} className="border-l-2 border-l-cyan pl-3 py-1">
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className={`text-[10px] font-bold ${
                      msg.type === 'ALERT' || msg.type === 'ERROR' ? 'text-red' :
                      msg.type === 'WARNING' ? 'text-yellow' : 'text-green'
                    }`}>{msg.type}</span>
                    <span className="text-[10px] text-text-muted">{msg.time}</span>
                  </div>
                  <p className="text-[11px] text-text-secondary leading-relaxed">{msg.text}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Section */}
      <div className="grid grid-cols-3 gap-4">
        {/* Holdings Table */}
        <div className="col-span-2 bg-bg-card rounded-lg border border-border">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider">Open Positions</h3>
            <div className="flex gap-4 text-[10px] text-text-muted">
              <span>POSITIONS: <span className="text-text-primary font-mono">{posCount}</span></span>
            </div>
          </div>
          {positions.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-text-muted">
              {error ? 'Unable to load positions' : 'No open positions'}
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="text-[10px] text-text-muted uppercase border-b border-border">
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-left px-4 py-2 font-medium">Direction</th>
                  <th className="text-right px-4 py-2 font-medium">QTY</th>
                  <th className="text-right px-4 py-2 font-medium">Entry</th>
                  <th className="text-right px-4 py-2 font-medium">Current</th>
                  <th className="text-right px-4 py-2 font-medium">P&L</th>
                  <th className="text-right px-4 py-2 font-medium">Stop</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => {
                  const pnl = p.unrealized_pnl ?? ((p.current_price ?? p.entry_price) - p.entry_price) * p.qty * (p.direction === 'short' ? -1 : 1)
                  const pnlPct = p.pnl_pct ?? (p.entry_price > 0 ? (pnl / (p.entry_price * p.qty)) * 100 : 0)
                  return (
                    <tr key={p.symbol} className="border-b border-border hover:bg-bg-card-hover transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-7 h-7 rounded-full bg-bg-card-hover flex items-center justify-center text-[10px] font-bold text-cyan">
                            {p.symbol.charAt(0)}
                          </div>
                          <span className="text-sm font-semibold text-text-primary">{p.symbol}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                          p.direction === 'long' ? 'bg-green/20 text-green' : 'bg-red/20 text-red'
                        }`}>
                          {p.direction.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">
                        {p.qty}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-secondary">
                        {formatCurrency(p.entry_price)}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">
                        {p.current_price != null ? formatCurrency(p.current_price) : '\u2014'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={`font-mono text-sm ${pnl >= 0 ? 'text-green' : 'text-red'}`}>
                          {formatCurrency(pnl)} ({formatPercent(pnlPct)})
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-muted">
                        {p.stop_price != null ? formatCurrency(p.stop_price) : '\u2014'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>

        {/* Execution Log */}
        <div className="col-span-1 bg-bg-card rounded-lg border border-border p-4">
          <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider mb-3">Execution Log</h3>
          {execLog.length === 0 ? (
            <p className="text-xs text-text-muted">No executions yet. Trades will appear here in real-time.</p>
          ) : (
            <div className="space-y-2">
              {execLog.map((log, i) => (
                <div key={i} className="flex items-center justify-between py-2 border-b border-border">
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                      log.side === 'LONG' || log.side === 'BUY' ? 'bg-green/20 text-green' :
                      log.side === 'SHORT' || log.side === 'SELL' ? 'bg-red/20 text-red' :
                      'bg-cyan/20 text-cyan'
                    }`}>
                      {log.side}
                    </span>
                    <span className="text-sm font-semibold text-text-primary">{log.ticker}</span>
                  </div>
                  <div className="text-right">
                    {log.price > 0 && (
                      <p className="font-mono text-xs text-text-primary">{formatCurrency(log.price)}</p>
                    )}
                    {log.shares > 0 && (
                      <p className="text-[10px] text-text-muted">{log.shares} shares</p>
                    )}
                    <p className="text-[10px] text-text-muted">{log.time}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
