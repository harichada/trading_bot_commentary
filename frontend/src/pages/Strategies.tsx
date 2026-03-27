import { useState, useCallback } from 'react'
import StatusBanner from '../components/StatusBanner'
import { usePolling } from '../hooks/usePolling'
import { idApi, type IdStrategy } from '../api/client'

export default function Strategies() {
  const [toggling, setToggling] = useState<string | null>(null)

  const { data: strategies, error, refresh } = usePolling<IdStrategy[]>(
    useCallback(() => idApi.getStrategies(), []),
    10000,
  )

  const handleToggle = async (strat: IdStrategy) => {
    setToggling(strat.strategy_id)
    try {
      if (strat.enabled) {
        await idApi.disableStrategy(strat.strategy_id)
      } else {
        await idApi.enableStrategy(strat.strategy_id)
      }
      refresh()
    } catch {
      // error handled by polling
    } finally {
      setToggling(null)
    }
  }

  const stratList = strategies ?? []
  const activeCount = stratList.filter(s => s.enabled).length
  const totalPnl = stratList.reduce((sum, s) => sum + (s.pnl_today ?? 0), 0)
  const totalTrades = stratList.reduce((sum, s) => sum + (s.trades_today ?? 0), 0)

  return (
    <div className="space-y-4">
      <StatusBanner label="Intraday Engine" error={error} wsConnected={true} onRetry={refresh} />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-text-primary">Intraday Strategy Manager</h1>
          <p className="text-xs text-text-muted mt-1">
            {stratList.length} strategies registered | {activeCount} active | {totalTrades} trades today
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={async () => { try { await idApi.start(); refresh() } catch {} }}
            className="px-4 py-2 text-xs font-semibold rounded-lg bg-green text-black hover:bg-green/90 transition-colors"
          >
            START ALL
          </button>
          <button
            onClick={async () => { try { await idApi.stop(); refresh() } catch {} }}
            className="px-4 py-2 text-xs font-semibold rounded-lg bg-red/10 text-red border border-red/30 hover:bg-red/20 transition-colors"
          >
            STOP ALL
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-bg-card rounded-lg border border-border p-4">
          <p className="text-[10px] text-text-muted uppercase">Active Strategies</p>
          <p className="font-mono text-xl font-semibold text-text-primary">{activeCount}</p>
        </div>
        <div className="bg-bg-card rounded-lg border border-border p-4">
          <p className="text-[10px] text-text-muted uppercase">Total Trades Today</p>
          <p className="font-mono text-xl font-semibold text-text-primary">{totalTrades}</p>
        </div>
        <div className="bg-bg-card rounded-lg border border-border p-4">
          <p className="text-[10px] text-text-muted uppercase">Combined P&L Today</p>
          <p className={`font-mono text-xl font-semibold ${totalPnl >= 0 ? 'text-green' : 'text-red'}`}>
            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
          </p>
        </div>
        <div className="bg-bg-card rounded-lg border border-border p-4">
          <p className="text-[10px] text-text-muted uppercase">Avg Win Rate</p>
          <p className="font-mono text-xl font-semibold text-text-primary">
            {stratList.length > 0
              ? `${(stratList.reduce((s, st) => s + (st.win_rate ?? 0), 0) / stratList.length).toFixed(1)}%`
              : '\u2014'}
          </p>
        </div>
      </div>

      {/* Strategy List */}
      <div className="bg-bg-card rounded-lg border border-border">
        <div className="px-4 py-3 border-b border-border">
          <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider">
            All Strategies
          </h3>
        </div>

        {stratList.length === 0 ? (
          <div className="px-4 py-8 text-center text-xs text-text-muted">
            {error ? 'Unable to load strategies from intraday engine' : 'No strategies registered'}
          </div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="text-[10px] text-text-muted uppercase border-b border-border">
                <th className="text-left px-4 py-2 font-medium">Strategy</th>
                <th className="text-center px-4 py-2 font-medium">Status</th>
                <th className="text-right px-4 py-2 font-medium">Trades Today</th>
                <th className="text-right px-4 py-2 font-medium">P&L Today</th>
                <th className="text-right px-4 py-2 font-medium">Win Rate</th>
                <th className="text-center px-4 py-2 font-medium">Enabled</th>
                <th className="text-right px-4 py-2 font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {stratList.map((strat) => (
                <tr key={strat.strategy_id} className="border-b border-border hover:bg-bg-card-hover transition-colors">
                  <td className="px-4 py-3">
                    <div>
                      <p className="text-sm font-semibold text-text-primary">
                        {strat.name ?? strat.strategy_id}
                      </p>
                      <p className="text-[10px] text-text-muted">{strat.strategy_id}</p>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                      strat.enabled
                        ? 'bg-green/10 text-green'
                        : 'bg-red/10 text-red'
                    }`}>
                      {strat.status?.toUpperCase() ?? (strat.enabled ? 'ACTIVE' : 'DISABLED')}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">
                    {strat.trades_today ?? 0}
                  </td>
                  <td className={`px-4 py-3 text-right font-mono text-sm ${
                    (strat.pnl_today ?? 0) >= 0 ? 'text-green' : 'text-red'
                  }`}>
                    {(strat.pnl_today ?? 0) >= 0 ? '+' : ''}${(strat.pnl_today ?? 0).toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-sm text-text-primary">
                    {strat.win_rate != null ? `${strat.win_rate.toFixed(1)}%` : '\u2014'}
                  </td>
                  <td className="px-4 py-3 text-center">
                    <button
                      onClick={() => handleToggle(strat)}
                      disabled={toggling === strat.strategy_id}
                      className={`w-10 h-5 rounded-full transition-colors relative ${
                        strat.enabled ? 'bg-green' : 'bg-bg-card-hover'
                      } ${toggling === strat.strategy_id ? 'opacity-50' : ''}`}
                    >
                      <span
                        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
                          strat.enabled ? 'translate-x-5' : 'translate-x-0.5'
                        }`}
                      />
                    </button>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => handleToggle(strat)}
                      disabled={toggling === strat.strategy_id}
                      className={`px-2 py-1 text-[10px] font-bold rounded transition-colors ${
                        strat.enabled
                          ? 'bg-red/10 text-red hover:bg-red/20'
                          : 'bg-green/10 text-green hover:bg-green/20'
                      }`}
                    >
                      {strat.enabled ? 'DISABLE' : 'ENABLE'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
