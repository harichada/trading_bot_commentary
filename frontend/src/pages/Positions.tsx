import { useState, useEffect, useCallback } from 'react'
import StatusBanner from '../components/StatusBanner'
import { usePolling } from '../hooks/usePolling'
import { useWebSocket } from '../hooks/useWebSocket'
import { useFormatters } from '../hooks/useFormatters'
import { gfApi, type GfState, type GfPosition, type GfTrade } from '../api/client'

type OrderTab = 'Active Positions' | 'Trade History'

export default function Positions() {
  const { formatCurrency, formatPercent } = useFormatters()
  const [orderTab, setOrderTab] = useState<OrderTab>('Active Positions')
  const [search, setSearch] = useState('')

  const { data: state, error, refresh } = usePolling<GfState>(
    useCallback(() => gfApi.getState(), []),
    10000,
  )

  const { lastMessage, connected: wsConnected } = useWebSocket('/gf-ws')

  const [positions, setPositions] = useState<GfPosition[]>([])
  const [trades, setTrades] = useState<GfTrade[]>([])

  useEffect(() => {
    if (state?.positions && Array.isArray(state.positions)) {
      setPositions(state.positions)
    }
    if (state?.trades && Array.isArray(state.trades)) {
      setTrades(state.trades)
    }
  }, [state])

  // Real-time WS updates
  useEffect(() => {
    if (!lastMessage) return
    if (lastMessage.type === 'positions_update' && Array.isArray(lastMessage.positions)) {
      setPositions(lastMessage.positions as GfPosition[])
    }
    if (lastMessage.type === 'trade') {
      const newTrades = Array.isArray(lastMessage.trades) ? lastMessage.trades as GfTrade[] : [lastMessage as unknown as GfTrade]
      setTrades(prev => [...newTrades, ...prev].slice(0, 200))
    }
  }, [lastMessage])

  const filteredPositions = positions.filter(p =>
    p.symbol.toLowerCase().includes(search.toLowerCase())
  )
  const filteredTrades = trades.filter(t =>
    t.symbol.toLowerCase().includes(search.toLowerCase())
  )

  const totalUnrealized = positions.reduce((sum, p) => {
    const pnl = p.unrealized_pnl ?? ((p.current_price ?? p.entry_price) - p.entry_price) * p.qty * (p.direction === 'short' ? -1 : 1)
    return sum + pnl
  }, 0)
  const totalCost = positions.reduce((sum, p) => sum + p.entry_price * p.qty, 0)
  const totalUnrealizedPct = totalCost > 0 ? (totalUnrealized / totalCost) * 100 : 0

  return (
    <div className="space-y-4">
      <StatusBanner label="Gap Fade Engine" error={error} wsConnected={wsConnected} onRetry={refresh} />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-bold text-text-primary">Rudra Trading Engine</h1>
          <span className="text-[10px] text-yellow font-semibold">EQUITIES</span>
          <span className="text-[10px] text-text-muted">{positions.length} ACTIVE</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-text-primary">
            {state?.equity != null ? formatCurrency(state.equity as number) : '\u2014'}
          </span>
        </div>
      </div>

      {/* Unrealized P&L Banner */}
      <div className="text-right">
        <span className="text-[10px] text-text-muted uppercase">Total Unrealized P&L</span>
        <p className={`font-mono text-lg font-bold ${totalUnrealized >= 0 ? 'text-green' : 'text-red'}`}>
          {positions.length > 0
            ? `${totalUnrealized >= 0 ? '+' : ''}${formatCurrency(totalUnrealized)}`
            : '\u2014'}
          {positions.length > 0 && (
            <span className="text-xs ml-1">({formatPercent(totalUnrealizedPct)})</span>
          )}
        </p>
      </div>

      {/* Search */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-xs">
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-text-muted absolute left-3 top-1/2 -translate-y-1/2">
            <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search Symbol..."
            className="w-full bg-bg-card border border-border rounded-lg pl-10 pr-4 py-2 text-xs text-text-primary placeholder:text-text-muted focus:outline-none focus:border-cyan"
          />
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-bg-card rounded-lg border border-border">
        <div className="flex items-center gap-0 border-b border-border">
          {(['Active Positions', 'Trade History'] as OrderTab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setOrderTab(tab)}
              className={`px-4 py-3 text-xs font-semibold transition-colors relative ${
                orderTab === tab
                  ? 'text-green'
                  : 'text-text-muted hover:text-text-secondary'
              }`}
            >
              {tab}
              {tab === 'Active Positions' && positions.length > 0 && (
                <span className="ml-1 text-[10px] bg-green/20 text-green px-1.5 rounded">{positions.length}</span>
              )}
              {tab === 'Trade History' && trades.length > 0 && (
                <span className="ml-1 text-[10px] bg-cyan/20 text-cyan px-1.5 rounded">{trades.length}</span>
              )}
              {orderTab === tab && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-green" />}
            </button>
          ))}
        </div>

        {orderTab === 'Active Positions' && (
          filteredPositions.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-text-muted">
              No open positions
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
                  <th className="text-right px-4 py-2 font-medium">Unrealized P&L</th>
                  <th className="text-right px-4 py-2 font-medium">Stop</th>
                  <th className="text-right px-4 py-2 font-medium">Target</th>
                </tr>
              </thead>
              <tbody>
                {filteredPositions.map((p) => {
                  const pnl = p.unrealized_pnl ?? ((p.current_price ?? p.entry_price) - p.entry_price) * p.qty * (p.direction === 'short' ? -1 : 1)
                  const pnlPct = p.pnl_pct ?? (p.entry_price > 0 ? (pnl / (p.entry_price * p.qty)) * 100 : 0)
                  return (
                    <tr key={p.symbol} className="border-b border-border hover:bg-bg-card-hover transition-colors">
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-8 h-8 rounded-full bg-bg-card-hover flex items-center justify-center text-[10px] font-bold text-cyan">
                            {p.symbol.charAt(0)}
                          </div>
                          <div>
                            <p className="text-sm font-semibold text-text-primary">{p.symbol}</p>
                            <p className="text-[10px] text-text-muted">{p.entry_time ?? ''}</p>
                          </div>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                          p.direction === 'long' ? 'bg-green/20 text-green' : 'bg-red/20 text-red'
                        }`}>
                          {p.direction.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">{p.qty}</td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-secondary">{formatCurrency(p.entry_price)}</td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">
                        {p.current_price != null ? formatCurrency(p.current_price) : '\u2014'}
                      </td>
                      <td className={`px-4 py-3 text-right font-mono text-sm ${pnl >= 0 ? 'text-green' : 'text-red'}`}>
                        {formatCurrency(pnl)} ({formatPercent(pnlPct)})
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-muted">
                        {p.stop_price != null ? formatCurrency(p.stop_price) : '\u2014'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-sm text-text-muted">
                        {p.target_price != null ? formatCurrency(p.target_price) : '\u2014'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )
        )}

        {orderTab === 'Trade History' && (
          filteredTrades.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-text-muted">
              No trade history
            </div>
          ) : (
            <table className="w-full">
              <thead>
                <tr className="text-[10px] text-text-muted uppercase border-b border-border">
                  <th className="text-left px-4 py-2 font-medium">Symbol</th>
                  <th className="text-left px-4 py-2 font-medium">Direction</th>
                  <th className="text-right px-4 py-2 font-medium">Entry</th>
                  <th className="text-right px-4 py-2 font-medium">Exit</th>
                  <th className="text-right px-4 py-2 font-medium">QTY</th>
                  <th className="text-right px-4 py-2 font-medium">P&L</th>
                  <th className="text-left px-4 py-2 font-medium">Exit Reason</th>
                  <th className="text-left px-4 py-2 font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {filteredTrades.map((t, i) => (
                  <tr key={i} className="border-b border-border hover:bg-bg-card-hover transition-colors">
                    <td className="px-4 py-3 text-sm font-semibold text-text-primary">{t.symbol}</td>
                    <td className="px-4 py-3">
                      <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                        t.direction === 'long' ? 'bg-green/20 text-green' : 'bg-red/20 text-red'
                      }`}>
                        {t.direction.toUpperCase()}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right font-mono text-sm text-text-secondary">{formatCurrency(t.entry_price)}</td>
                    <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">{formatCurrency(t.exit_price)}</td>
                    <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">{t.qty}</td>
                    <td className={`px-4 py-3 text-right font-mono text-sm ${t.pnl >= 0 ? 'text-green' : 'text-red'}`}>
                      {formatCurrency(t.pnl)}
                    </td>
                    <td className="px-4 py-3 text-xs text-text-muted">{t.exit_reason ?? '\u2014'}</td>
                    <td className="px-4 py-3 text-xs text-text-muted font-mono">{t.exit_time ?? '\u2014'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        )}
      </div>

      {/* Trading Status Widget */}
      <div className="fixed bottom-4 right-4 bg-bg-card rounded-lg border border-border p-3 shadow-lg z-50">
        <div className="flex items-center gap-2 mb-1">
          <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green animate-pulse' : 'bg-red'}`} />
          <span className="text-xs font-semibold text-text-primary">TRADING STATUS</span>
        </div>
        <p className="text-[10px] text-text-muted">
          WS: {wsConnected ? 'Connected' : 'Disconnected'} | Positions: {positions.length}
        </p>
      </div>
    </div>
  )
}
