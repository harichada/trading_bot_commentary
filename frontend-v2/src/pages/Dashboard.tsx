import { useState, useEffect, useMemo } from 'react'
import { useTraderState } from '../hooks/useTraderState'
import { api } from '../lib/api'
import type { Position, Trade } from '../lib/api'
import EquityChart from '../components/EquityChart'
import styles from './Dashboard.module.css'

const TIME_PERIODS = ['1D', '1W', '1M', '1Y', 'YTD', 'ALL'] as const

const NEWS_HEADLINES = [
  'US strikes 11,000 Iranian-linked targets in Syria',
  'Saudi Arabia ramps oil output amid OPEC tensions',
  'Dow Jones enters correction territory, down 8%',
  'Nvidia resumes chip exports to China after waiver',
  'Fed signals two rate cuts likely before year-end',
  'Treasury yields spike as inflation data surprises',
]

/* ── Donut Chart (SVG) ── */
function DonutChart({ value, total, size = 140 }: { value: number; total: number; size?: number }) {
  const r = (size - 20) / 2
  const circ = 2 * Math.PI * r
  const pct = total > 0 ? value / total : 0
  const offset = circ * (1 - pct)
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {/* Background track */}
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={10} />
      {/* Progress arc */}
      <circle cx={size/2} cy={size/2} r={r} fill="none" stroke="#00FFBB" strokeWidth={10}
        strokeDasharray={`${circ}`} strokeDashoffset={offset}
        strokeLinecap="round" transform={`rotate(-90 ${size/2} ${size/2})`}
        style={{ transition: 'stroke-dashoffset 0.8s ease' }} />
      {/* Center value */}
      <text x={size/2} y={size/2 + 2} textAnchor="middle" fill="#fff" fontSize={16} fontWeight={700}
        fontFamily="'JetBrains Mono', monospace">
        ${value >= 1000 ? `${(value / 1000).toFixed(0)}K` : value.toFixed(0)}
      </text>
    </svg>
  )
}

/* ── Position Row ── */
function PositionRow({ p }: { p: Position }) {
  const pnlPos = p.pnl >= 0
  return (
    <tr>
      <td><span className={`${styles.statusDot} ${pnlPos ? styles.dotGreen : styles.dotRed}`} /></td>
      <td className={styles.cellTicker}>{p.direction.toUpperCase()}</td>
      <td className={styles.cellTicker}>{p.symbol}</td>
      <td className={styles.cellMuted}>{p.strategy_id || '—'}</td>
      <td className={styles.cellMono}>{p.entry_time}</td>
      <td className={styles.cellMono}>${(p.entry_price * p.remaining_shares).toFixed(0)}</td>
      <td className={styles.cellMono}>{p.remaining_shares}</td>
      <td className={styles.cellMono}>${p.entry_price.toFixed(2)}</td>
      <td className={styles.cellMono}>${p.current_price.toFixed(2)}</td>
      <td className={`${styles.cellMono} ${pnlPos ? styles.pnlUp : styles.pnlDown}`}>
        {pnlPos ? '+' : ''}${p.pnl.toFixed(2)}
      </td>
    </tr>
  )
}

/* ══════════════════════════════════════════════
   DASHBOARD
   ══════════════════════════════════════════════ */
export default function Dashboard() {
  const { state, health } = useTraderState(3000)
  const [trades, setTrades] = useState<Trade[]>([])
  const [period, setPeriod] = useState<string>('ALL')
  // controls always visible

  useEffect(() => {
    api.trades().then(setTrades).catch(() => {})
  }, [])

  const equity = state?.equity ?? 0
  const dailyPnl = state?.daily_pnl ?? 0
  const dailyPnlPct = state?.daily_pnl_pct ?? 0
  const positions = state?.positions ?? []
  const messages = state?.messages ?? []
  const unrealizedPnl = positions.reduce((s, p) => s + (p.pnl ?? 0), 0)

  // Circuit breakers from state
  const daily = (state as Record<string, unknown> | null)?.daily_stats as Record<string, unknown> | undefined
  const cbDaily = Number(daily?.pnl ?? 0)
  const cbConsec = Number(daily?.consecutive_losses ?? 0)
  const cbHalted = daily?.halted === true

  // Build equity curve from trades
  const curve = useMemo(() => {
    if (!trades.length) return { dates: [] as string[], equity: [] as number[] }
    const startEquity = equity - trades.reduce((s, t) => s + t.pnl, 0)
    const sorted = [...trades].sort((a, b) => a.entry_time.localeCompare(b.entry_time))
    const dates: string[] = ['Start']
    const equities: number[] = [startEquity]
    let running = startEquity
    for (const t of sorted) {
      running += t.pnl
      dates.push(t.exit_time?.split(' ')[0] ?? t.entry_time.split(' ')[0])
      equities.push(running)
    }
    return { dates, equity: equities }
  }, [trades, equity])

  const pnlPositive = dailyPnl >= 0

  return (
    <div className={styles.page}>
      {/* ── Greeting ── */}
      <h1 className={styles.greeting}>Hi Hari Kishore!</h1>

      {/* ── Circuit Breakers ── */}
      <div className={styles.cbRow}>
        <span className={`${styles.cbDot} ${cbHalted ? styles.cbRed : cbDaily < -500 ? styles.cbYellow : styles.cbGreen}`} />
        <span className={styles.cbLabel}>Daily: ${cbDaily.toFixed(0)}</span>
        <span className={`${styles.cbDot} ${cbConsec >= 3 ? styles.cbRed : cbConsec >= 2 ? styles.cbYellow : styles.cbGreen}`} />
        <span className={styles.cbLabel}>Consec: {cbConsec}</span>
        <span className={`${styles.cbDot} ${cbHalted ? styles.cbRed : styles.cbGreen}`} />
        <span className={styles.cbLabel}>{cbHalted ? 'HALTED' : 'OK'}</span>
      </div>

      {/* ── Top Row: Balance | Allocation | News ── */}
      <div className={styles.topRow}>
        {/* Balance Card */}
        <div className={`${styles.card} ${styles.balanceCard}`}>
          <div className={styles.balanceHeader}>
            <span className={styles.balanceLabel}>Total Balance</span>
            <div className={styles.periodToggle}>
              {TIME_PERIODS.map(p => (
                <button key={p}
                  className={`${styles.periodBtn} ${period === p ? styles.periodActive : ''}`}
                  onClick={() => setPeriod(p)}
                >{p}</button>
              ))}
            </div>
          </div>
          <div className={styles.balanceValue}>
            ${equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </div>
          <div className={styles.balanceChange}>
            <span className={`${styles.changeBadge} ${pnlPositive ? styles.changeUp : styles.changeDown}`}>
              {pnlPositive ? '+' : ''}{(dailyPnlPct * 100).toFixed(2)}%
            </span>
            <span className={pnlPositive ? styles.pnlUp : styles.pnlDown}>
              ({pnlPositive ? '+' : ''}${dailyPnl.toFixed(2)})
            </span>
          </div>
          <div className={styles.chartWrap}>
            {curve.equity.length > 1 ? (
              <EquityChart dates={curve.dates} equity={curve.equity} height={180} />
            ) : (
              <div className={styles.chartEmpty}>
                {trades.length === 0 ? 'Loading chart...' : 'No trade data yet'}
              </div>
            )}
          </div>
        </div>

        {/* Allocation Card */}
        <div className={`${styles.card} ${styles.allocCard}`}>
          <span className={styles.cardLabel}>Allocation</span>
          <div className={styles.donutWrap}>
            <DonutChart value={equity} total={equity || 100000} />
          </div>
          <button className={styles.manageBtn}>Manage strategies</button>
        </div>

        {/* Bot Feed + News Card */}
        <div className={`${styles.card} ${styles.newsCard}`}>
          {/* Live Bot Feed */}
          <span className={styles.cardLabel}>Bot Activity</span>
          <div className={styles.botFeed}>
            {messages.slice(-8).reverse().map((msg, i) => (
              <div key={i} className={styles.feedLine}>
                <span className={styles.feedTime}>{msg.time}</span>
                <span className={styles.feedType}>{msg.type}</span>
                <span className={styles.feedText}>{msg.text}</span>
              </div>
            ))}
            {messages.length === 0 && <div className={styles.feedEmpty}>Waiting for activity...</div>}
          </div>

          {/* Divider */}
          <div className={styles.feedDivider} />

          <span className={styles.cardLabel}>Recent News</span>
          <div className={styles.newsList}>
            {NEWS_HEADLINES.map((h, i) => (
              <div key={i} className={styles.newsItem}>
                <span className={styles.newsDot} />
                <span>{h}</span>
              </div>
            ))}
          </div>
          <button className={styles.newsBtn}>Chat with news</button>
        </div>
      </div>

      {/* ── Middle Row: Available | Connect ── */}
      <div className={styles.midRow}>
        <div className={`${styles.card} ${styles.availCard}`}>
          <span className={styles.cardLabel}>Available</span>
          <div className={styles.availValue}>
            ${equity.toLocaleString('en-US', { maximumFractionDigits: 0 })}
          </div>
          <button className={styles.allocateBtn}>Allocate</button>
        </div>
        <div className={`${styles.card} ${styles.connectCard}`}>
          <div>
            <div className={styles.connectTitle}>Run your strategies on your portfolio</div>
            <div className={styles.connectSub}>Connect your broker to start automated trading</div>
          </div>
          <div className={styles.connectRight}>
            <div className={styles.brokerIcons}>
              <span className={styles.brokerIcon}>A</span>
              <span className={styles.brokerIcon}>IB</span>
            </div>
            <button className={styles.connectBtn}>
              {health?.trader_status === 'stopped' ? 'Connect' : 'Connected'}
            </button>
          </div>
        </div>
      </div>

      {/* ── Bot Controls ── */}
      <div className={styles.controlsBar}>
        <button className={styles.ctrlBtn} onClick={() => api.start()}>▶ Start</button>
        <button className={styles.ctrlBtn} onClick={() => api.scan()}>Scan</button>
        <button className={styles.ctrlBtn} onClick={() => api.pause()}>Pause</button>
        <button className={styles.ctrlBtn} onClick={() => api.resume()}>Resume</button>
        <button className={`${styles.ctrlBtn} ${styles.ctrlDanger}`} onClick={() => api.stop()}>■ Stop</button>
      </div>

      {/* ── Open Trades Table ── */}
      <div className={styles.card}>
        <div className={styles.tradesHeader}>
          <span className={styles.tradesTitle}>Open Trades ({positions.length})</span>
          <span className={`${styles.unrealized} ${unrealizedPnl >= 0 ? styles.pnlUp : styles.pnlDown}`}>
            Unrealized profits: {unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}
          </span>
        </div>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead>
              <tr>
                <th>Status</th>
                <th>Position</th>
                <th>Asset</th>
                <th>Strategy</th>
                <th>Entry date</th>
                <th>Size</th>
                <th>Qty</th>
                <th>Entry price</th>
                <th>Current price</th>
                <th>PNL</th>
              </tr>
            </thead>
            <tbody>
              {positions.length === 0 ? (
                <tr><td colSpan={10} className={styles.emptyRow}>No position</td></tr>
              ) : (
                positions.map(p => <PositionRow key={p.symbol} p={p} />)
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
