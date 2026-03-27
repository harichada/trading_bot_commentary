import { useState, useEffect, useCallback } from 'react'
import Chart from '../components/Chart'
import PositionCard from '../components/PositionCard'
import StatusBanner from '../components/StatusBanner'
import { usePolling } from '../hooks/usePolling'
import { useWebSocket } from '../hooks/useWebSocket'
import { useFormatters } from '../hooks/useFormatters'
import { gfApi, type GfState, type GfPosition } from '../api/client'

export default function Charting() {
  const { formatCurrency } = useFormatters()
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null)
  const [searchTicker, setSearchTicker] = useState('')

  const { data: state, error, refresh } = usePolling<GfState>(
    useCallback(() => gfApi.getState(), []),
    10000,
  )

  const { lastMessage, connected: wsConnected } = useWebSocket('/gf-ws')

  const [positions, setPositions] = useState<GfPosition[]>([])

  useEffect(() => {
    if (state?.positions && Array.isArray(state.positions)) {
      setPositions(state.positions)
    }
  }, [state])

  useEffect(() => {
    if (lastMessage?.type === 'positions_update' && Array.isArray(lastMessage.positions)) {
      setPositions(lastMessage.positions as GfPosition[])
    }
  }, [lastMessage])

  // Auto-select first position
  useEffect(() => {
    if (!selectedSymbol && positions.length > 0) {
      setSelectedSymbol(positions[0].symbol)
    }
  }, [positions, selectedSymbol])

  const totalUnrealized = positions.reduce((sum, p) => {
    const pnl = p.unrealized_pnl ?? ((p.current_price ?? p.entry_price) - p.entry_price) * p.qty * (p.direction === 'short' ? -1 : 1)
    return sum + pnl
  }, 0)

  return (
    <div className="space-y-4">
      <StatusBanner label="Gap Fade Engine" error={error} wsConnected={wsConnected} onRetry={refresh} />

      {/* Search Bar */}
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-text-muted absolute left-3 top-1/2 -translate-y-1/2">
            <path fillRule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clipRule="evenodd" />
          </svg>
          <input
            type="text"
            value={searchTicker}
            onChange={(e) => setSearchTicker(e.target.value)}
            placeholder="Search Ticker (AAPL, NVDA...) to load chart"
            className="w-full bg-bg-card border border-border rounded-lg pl-10 pr-4 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:outline-none focus:border-cyan"
          />
        </div>
      </div>

      {/* Chart + Positions */}
      <div className="grid grid-cols-4 gap-4">
        {/* Chart Area */}
        <div className="col-span-3 bg-bg-card rounded-lg border border-border p-4">
          <div className="flex items-center gap-4 mb-3">
            <h2 className="text-lg font-bold text-text-primary">
              {selectedSymbol ?? 'Select a position'}
            </h2>
            {selectedSymbol && positions.find(p => p.symbol === selectedSymbol) && (
              <>
                <span className="text-text-muted text-sm">
                  {positions.find(p => p.symbol === selectedSymbol)?.direction.toUpperCase()}
                </span>
                {positions.find(p => p.symbol === selectedSymbol)?.current_price != null && (
                  <span className="font-mono text-lg font-semibold text-text-primary">
                    {formatCurrency(positions.find(p => p.symbol === selectedSymbol)!.current_price!)}
                  </span>
                )}
              </>
            )}
          </div>

          <div className="text-[10px] text-text-muted mb-2">
            Chart shows generated data. Real-time price bars require a market data feed.
          </div>

          <Chart type="candlestick" height={420} showVolume={true} />
        </div>

        {/* Active Positions Sidebar */}
        <div className="col-span-1 bg-bg-card rounded-lg border border-border">
          <div className="px-4 py-3 border-b border-border flex items-center justify-between">
            <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider">Active Positions</h3>
            <span className={`text-[10px] font-bold ${positions.length > 0 ? 'text-green' : 'text-text-muted'}`}>
              {positions.length} ACTIVE
            </span>
          </div>

          {positions.length === 0 ? (
            <div className="px-4 py-8 text-center text-xs text-text-muted">
              No open positions
            </div>
          ) : (
            <div className="divide-y divide-border">
              {positions.map((pos) => {
                const pnl = pos.unrealized_pnl ?? ((pos.current_price ?? pos.entry_price) - pos.entry_price) * pos.qty * (pos.direction === 'short' ? -1 : 1)
                const pnlPct = pos.pnl_pct ?? (pos.entry_price > 0 ? (pnl / (pos.entry_price * pos.qty)) * 100 : 0)
                return (
                  <div
                    key={pos.symbol}
                    onClick={() => setSelectedSymbol(pos.symbol)}
                    className={`cursor-pointer ${selectedSymbol === pos.symbol ? 'bg-bg-card-hover' : ''}`}
                  >
                    <PositionCard
                      symbol={pos.symbol}
                      shares={pos.qty}
                      avgPrice={pos.entry_price}
                      currentPrice={pos.current_price ?? pos.entry_price}
                      pnl={pnl}
                      pnlPercent={pnlPct}
                    />
                  </div>
                )
              })}
            </div>
          )}

          <div className="px-4 py-3 border-t border-border space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-text-muted">TOTAL UNREALIZED P&L</span>
              <span className={`font-mono font-semibold ${totalUnrealized >= 0 ? 'text-green' : 'text-red'}`}>
                {positions.length > 0 ? formatCurrency(totalUnrealized) : '\u2014'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Status Bar */}
      <div className="flex items-center justify-between px-4 py-2 bg-bg-card rounded-lg border border-border text-[10px]">
        <div className="flex items-center gap-4">
          <span className="text-text-muted">PRICE: {wsConnected ? 'LIVE' : 'STALE'}</span>
          <span className="text-text-muted">POSITIONS: {positions.length}</span>
        </div>
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-1 ${wsConnected ? 'text-green' : 'text-red'}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? 'bg-green' : 'bg-red'}`} />
            WS: {wsConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </span>
          <span className="text-text-muted">TIME: {new Date().toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  )
}
