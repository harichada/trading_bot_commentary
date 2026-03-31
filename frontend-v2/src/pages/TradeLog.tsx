import { useState, useEffect, useMemo } from 'react'
import { api, type Trade } from '../lib/api'
import PnlHistogram from '../components/PnlHistogram'
import WinRateGauge from '../components/WinRateGauge'
import ui from '../components/ui.module.css'
import styles from './TradeLog.module.css'

const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function computeAnalytics(trades: Trade[]) {
  if (trades.length === 0) return null
  const wins = trades.filter(t => t.pnl > 0)
  const losses = trades.filter(t => t.pnl <= 0)
  const best = trades.reduce((a, b) => a.pnl > b.pnl ? a : b)
  const worst = trades.reduce((a, b) => a.pnl < b.pnl ? a : b)
  const avgWinner = wins.length > 0 ? wins.reduce((s, t) => s + t.pnl, 0) / wins.length : 0
  const avgLoser = losses.length > 0 ? losses.reduce((s, t) => s + t.pnl, 0) / losses.length : 0

  // Streaks
  let maxWin = 0, maxLoss = 0, curWin = 0, curLoss = 0
  for (const t of trades) {
    if (t.pnl > 0) { curWin++; curLoss = 0; maxWin = Math.max(maxWin, curWin) }
    else { curLoss++; curWin = 0; maxLoss = Math.max(maxLoss, curLoss) }
  }

  // P&L by day of week
  const byDay: Record<number, number> = {}
  for (const t of trades) {
    const d = new Date(t.entry_time).getDay()
    byDay[d] = (byDay[d] ?? 0) + t.pnl
  }

  return { best, worst, avgWinner, avgLoser, maxWin, maxLoss, byDay }
}

export default function TradeLog() {
  const [trades, setTrades] = useState<Trade[]>([])
  const [loading, setLoading] = useState(true)

  // Filters
  const [filterSymbol, setFilterSymbol] = useState('')
  const [filterDir, setFilterDir] = useState('')
  const [filterStrategy, setFilterStrategy] = useState('')
  const [sortCol, setSortCol] = useState<string>('entry_time')
  const [sortAsc, setSortAsc] = useState(false)

  useEffect(() => {
    api.trades().then(t => { setTrades(t); setLoading(false) }).catch(() => setLoading(false))
  }, [])

  const filtered = useMemo(() => {
    let result = [...trades]
    if (filterSymbol) result = result.filter(t => t.symbol.toUpperCase().includes(filterSymbol.toUpperCase()))
    if (filterDir) result = result.filter(t => t.side === filterDir)
    if (filterStrategy) result = result.filter(t => t.strategy_id.includes(filterStrategy))

    result.sort((a, b) => {
      const av = (a as Record<string, unknown>)[sortCol]
      const bv = (b as Record<string, unknown>)[sortCol]
      if (typeof av === 'number' && typeof bv === 'number') return sortAsc ? av - bv : bv - av
      return sortAsc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av))
    })
    return result
  }, [trades, filterSymbol, filterDir, filterStrategy, sortCol, sortAsc])

  const analytics = useMemo(() => computeAnalytics(filtered), [filtered])

  const toggleSort = (col: string) => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
  }

  const thCls = (col: string) =>
    sortCol === col ? (sortAsc ? ui.sortedAsc ?? '' : ui.sortedDesc ?? '') : ''

  const strategies = useMemo(() => [...new Set(trades.map(t => t.strategy_id).filter(Boolean))], [trades])

  const maxDayPnl = analytics ? Math.max(...Object.values(analytics.byDay).map(Math.abs), 1) : 1

  return (
    <div className={styles.layout}>
      {/* Left: Filters + Table */}
      <div className={styles.main}>
        {/* Filter Bar */}
        <div className={ui.filterBar}>
          <div className={ui.filterGroup}>
            <label className={ui.formLabel}>Symbol</label>
            <input
              className={ui.input}
              placeholder="e.g. TSLA"
              style={{ width: 100 }}
              value={filterSymbol}
              onChange={e => setFilterSymbol(e.target.value)}
            />
          </div>
          <div className={ui.filterGroup}>
            <label className={ui.formLabel}>Direction</label>
            <select className={ui.select} style={{ width: 100 }} value={filterDir} onChange={e => setFilterDir(e.target.value)}>
              <option value="">Both</option>
              <option value="short">Short</option>
              <option value="long">Long</option>
            </select>
          </div>
          <div className={ui.filterGroup}>
            <label className={ui.formLabel}>Strategy</label>
            <select className={ui.select} style={{ width: 160 }} value={filterStrategy} onChange={e => setFilterStrategy(e.target.value)}>
              <option value="">All Strategies</option>
              {strategies.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
          </div>
          <div className={ui.filterActions}>
            <span className="text-caption">{filtered.length} trades</span>
          </div>
        </div>

        {/* Table */}
        <div className={ui.card}>
          <div className={ui.tableScroll} style={{ maxHeight: 'calc(100vh - 280px)', overflowY: 'auto' }}>
            <table className={ui.dataTable}>
              <thead>
                <tr>
                  <th className={thCls('entry_time')} onClick={() => toggleSort('entry_time')}>Date</th>
                  <th className={thCls('symbol')} onClick={() => toggleSort('symbol')}>Ticker</th>
                  <th>Dir</th>
                  <th className={thCls('shares')} onClick={() => toggleSort('shares')}>Qty</th>
                  <th className={thCls('entry_price')} onClick={() => toggleSort('entry_price')}>Entry</th>
                  <th className={thCls('exit_price')} onClick={() => toggleSort('exit_price')}>Exit</th>
                  <th className={thCls('pnl')} onClick={() => toggleSort('pnl')}>P&L</th>
                  <th className={thCls('pnl_pct')} onClick={() => toggleSort('pnl_pct')}>P&L%</th>
                  <th>Hold</th>
                  <th>Strategy</th>
                  <th>Exit</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr className={ui.emptyRow}><td colSpan={11}>Loading trades...</td></tr>
                )}
                {!loading && filtered.length === 0 && (
                  <tr className={ui.emptyRow}><td colSpan={11}>No trades found</td></tr>
                )}
                {filtered.map(t => (
                  <tr key={`${t.trade_id}-${t.symbol}-${t.entry_time}`} className={t.pnl >= 0 ? ui.rowProfit : ui.rowLoss}>
                    <td className={ui.tdMuted}>{t.entry_time}</td>
                    <td className={ui.tdTicker}>{t.symbol}</td>
                    <td>
                      <span className={`${ui.badge} ${t.side === 'short' ? ui.badgeLoss : ui.badgeProfit} ${ui.badgeSm}`}>
                        {t.side.toUpperCase()}
                      </span>
                    </td>
                    <td>{t.shares}</td>
                    <td>${t.entry_price.toFixed(2)}</td>
                    <td>${t.exit_price.toFixed(2)}</td>
                    <td className={t.pnl >= 0 ? ui.tdProfit : ui.tdLoss}>
                      {t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}
                    </td>
                    <td className={t.pnl >= 0 ? ui.tdProfit : ui.tdLoss}>
                      {(t.pnl_pct * 100).toFixed(2)}%
                    </td>
                    <td className={ui.tdMuted}>{t.holding_minutes ?? '—'}m</td>
                    <td className={ui.tdMuted}>{t.strategy_id || '—'}</td>
                    <td className={ui.tdMuted}>{t.exit_reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Right: Analytics */}
      <div className={styles.analytics}>
        {/* Trade Analytics */}
        <div className={ui.card} style={{ padding: 'var(--space-5)' }}>
          <div className="section-title" style={{ marginBottom: 'var(--space-3)' }}>Trade Analytics</div>
          {analytics ? (
            <div className={ui.analyticsGrid}>
              <div className={ui.analyticsStat}>
                <span className={ui.analyticsLabel}>Best Trade</span>
                <span className={`${ui.analyticsValue} text-profit`}>+${analytics.best.pnl.toFixed(2)}</span>
                <span className={ui.analyticsSub}>{analytics.best.symbol}</span>
              </div>
              <div className={ui.analyticsStat}>
                <span className={ui.analyticsLabel}>Worst Trade</span>
                <span className={`${ui.analyticsValue} text-loss`}>${analytics.worst.pnl.toFixed(2)}</span>
                <span className={ui.analyticsSub}>{analytics.worst.symbol}</span>
              </div>
              <div className={ui.analyticsStat}>
                <span className={ui.analyticsLabel}>Avg Winner</span>
                <span className={`${ui.analyticsValue} text-profit`}>+${analytics.avgWinner.toFixed(2)}</span>
              </div>
              <div className={ui.analyticsStat}>
                <span className={ui.analyticsLabel}>Avg Loser</span>
                <span className={`${ui.analyticsValue} text-loss`}>${analytics.avgLoser.toFixed(2)}</span>
              </div>
              <div className={ui.analyticsStat}>
                <span className={ui.analyticsLabel}>Win Streak</span>
                <span className={ui.analyticsValue}>{analytics.maxWin}W</span>
              </div>
              <div className={ui.analyticsStat}>
                <span className={ui.analyticsLabel}>Loss Streak</span>
                <span className={`${ui.analyticsValue} text-loss`}>{analytics.maxLoss}L</span>
              </div>
            </div>
          ) : (
            <div className="text-muted" style={{ textAlign: 'center', padding: 'var(--space-4)' }}>No data</div>
          )}
        </div>

        {/* P&L by Day */}
        <div className={ui.card} style={{ padding: 'var(--space-5)' }}>
          <div className="section-title" style={{ marginBottom: 'var(--space-3)' }}>By Day of Week</div>
          {analytics && [1, 2, 3, 4, 5].map(d => {
            const pnl = analytics.byDay[d] ?? 0
            const pct = Math.abs(pnl) / maxDayPnl * 100
            return (
              <div key={d} className={ui.dayRow}>
                <span className={ui.dayLabel}>{DAYS[d]}</span>
                <div className={ui.dayBarWrap}>
                  <div
                    className={`${ui.dayBar} ${pnl < 0 ? ui.dayBarNeg : ''}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className={`${ui.dayPnl} ${pnl >= 0 ? 'text-profit' : 'text-loss'}`}>
                  {pnl >= 0 ? '+' : ''}${pnl.toFixed(0)}
                </span>
              </div>
            )
          })}
        </div>

        {/* P&L Distribution */}
        <div className={ui.card} style={{ padding: 'var(--space-4)' }}>
          <div className="section-title" style={{ marginBottom: 'var(--space-3)' }}>P&L Distribution</div>
          <PnlHistogram pnlValues={filtered.map(t => t.pnl)} />
        </div>

        {/* Win Rate Gauge */}
        {filtered.length > 0 && (
          <div className={ui.card} style={{ padding: 'var(--space-4)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 'var(--space-2)' }}>
            <div className="section-title">Win Rate</div>
            <WinRateGauge winRate={parseFloat((filtered.filter(t => t.pnl > 0).length / filtered.length * 100).toFixed(1))} />
          </div>
        )}
      </div>
    </div>
  )
}
