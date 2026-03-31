import { useState, useEffect } from 'react'
import { useToast } from '../components/Toast'
import s from './IntradayLab.module.css'

const LAB = '/api/lab'

interface StratParam { key: string; value: unknown; type: string }
interface StratResult { strategy_id: string; strategy_name: string; total_trades: number; win_rate: number; net_pnl: number; profit_factor: number; sharpe: number; max_drawdown_pct: number; avg_win: number; avg_loss: number; avg_hold_min: number; recommendation?: string; trades?: Record<string, unknown>[] }

function StatusBadge({ pf, sharpe }: { pf: number; sharpe: number }) {
  if (pf >= 1.5 && sharpe >= 2) return <span className={s.badgeReady}>✅ Ready</span>
  if (pf >= 1.0) return <span className={s.badgeWarn}>⚠️ Marginal</span>
  return <span className={s.badgeReject}>❌ Reject</span>
}

export default function IntradayLab() {
  const [strategies, setStrategies] = useState<{ id: string; params: Record<string, unknown> }[]>([])
  const [selected, setSelected] = useState('')
  const [params, setParams] = useState<Record<string, unknown>>({})
  const [sweepGrid, setSweepGrid] = useState<Record<string, string>>({}) // key -> "val1,val2,val3"
  const [sweepEnabled, setSweepEnabled] = useState<Set<string>>(new Set())

  // Run config
  const [startDate, setStartDate] = useState('2025-06-01')
  const [endDate, setEndDate] = useState('2026-03-15')
  const [capital, setCapital] = useState(100000)
  const [maxSymbols, setMaxSymbols] = useState(20)
  const [maxPositions, setMaxPositions] = useState(3)
  const [riskPct, setRiskPct] = useState(0.01)
  const [mode, setMode] = useState<'single' | 'sweep' | 'rank'>('single')

  // Results
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState({ pct: 0, msg: '' })
  const [result, setResult] = useState<StratResult | null>(null)
  const [sweepResults, setSweepResults] = useState<Record<string, unknown>[] | null>(null)
  const [rankings, setRankings] = useState<StratResult[]>([])

  const { showToast } = useToast()

  // Load strategies
  useEffect(() => {
    fetch(`${LAB}/strategies`).then(r => r.json())
      .then(d => { setStrategies(d.strategies ?? []); if (d.strategies?.[0]) selectStrategy(d.strategies[0].id, d.strategies) })
      .catch(() => {})
  }, [])

  // Poll when running
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => {
      fetch(`${LAB}/status`).then(r => r.json()).then(d => {
        setProgress({ pct: d.progress ?? 0, msg: d.message ?? '' })
        if (d.status === 'complete') {
          setRunning(false)
          fetch(`${LAB}/results`).then(r => r.json()).then(res => {
            if (res.rankings) setRankings(res.rankings)
            else if (res.results) setSweepResults(res.results)
            else setResult(res)
            showToast('Complete', 'success')
          })
        }
        if (d.status === 'error') { setRunning(false); showToast(`Error: ${d.message}`, 'error') }
      })
    }, 2000)
    return () => clearInterval(timer)
  }, [running])

  const selectStrategy = (id: string, strats?: typeof strategies) => {
    setSelected(id)
    const s = (strats ?? strategies).find(x => x.id === id)
    if (s) {
      setParams({ ...s.params })
      setSweepGrid({})
      setSweepEnabled(new Set())
    }
  }

  const updateParam = (key: string, val: string) => {
    const orig = strategies.find(x => x.id === selected)?.params[key]
    if (typeof orig === 'boolean') setParams(p => ({ ...p, [key]: val === 'true' }))
    else if (typeof orig === 'number' && Number.isInteger(orig)) setParams(p => ({ ...p, [key]: parseInt(val) || 0 }))
    else if (typeof orig === 'number') setParams(p => ({ ...p, [key]: parseFloat(val) || 0 }))
    else setParams(p => ({ ...p, [key]: val }))
  }

  const toggleSweep = (key: string) => {
    const next = new Set(sweepEnabled)
    if (next.has(key)) next.delete(key)
    else { next.add(key); if (!sweepGrid[key]) setSweepGrid(g => ({ ...g, [key]: String(params[key]) })) }
    setSweepEnabled(next)
  }

  const runSingle = () => {
    setRunning(true); setResult(null); setSweepResults(null)
    fetch(`${LAB}/backtest`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strategy_id: selected, config: params, start_date: startDate, end_date: endDate, capital, max_symbols_per_day: maxSymbols, max_positions: maxPositions, risk_pct: riskPct }),
    })
  }

  const runSweep = () => {
    const grid: Record<string, number[]> = {}
    for (const key of sweepEnabled) {
      const vals = (sweepGrid[key] ?? '').split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v))
      if (vals.length > 0) grid[key] = vals
    }
    if (Object.keys(grid).length === 0) { showToast('Enable at least one parameter for sweep', 'warning'); return }
    const combos = Object.values(grid).reduce((a, b) => a * b.length, 1)
    if (combos > 500) { showToast(`${combos} combinations — reduce grid size`, 'warning'); return }

    setRunning(true); setSweepResults(null); setResult(null)
    fetch(`${LAB}/sweep`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ strategy_id: selected, param_grid: grid, start_date: startDate, end_date: endDate, capital, max_symbols_per_day: maxSymbols, max_positions: maxPositions, risk_pct: riskPct }),
    })
  }

  const runRankAll = () => {
    setRunning(true); setRankings([])
    fetch(`${LAB}/rank-all`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ start_date: startDate, end_date: endDate, capital, max_symbols_per_day: maxSymbols }),
    })
  }

  const combos = [...sweepEnabled].reduce((a, key) => {
    const vals = (sweepGrid[key] ?? '').split(',').filter(v => v.trim())
    return a * Math.max(vals.length, 1)
  }, 1)

  return (
    <div className={s.page}>
      {/* Header */}
      <div className={s.header}>
        <div><h1 className={s.title}>Intraday Strategy Lab</h1><p className={s.sub}>Backtest, sweep, and rank intraday strategies on 203M 1-min bars</p></div>
      </div>

      {/* Mode tabs */}
      <div className={s.modeTabs}>
        <button className={`${s.modeTab} ${mode === 'single' ? s.modeActive : ''}`} onClick={() => setMode('single')}>Single Backtest</button>
        <button className={`${s.modeTab} ${mode === 'sweep' ? s.modeActive : ''}`} onClick={() => setMode('sweep')}>Parameter Sweep</button>
        <button className={`${s.modeTab} ${mode === 'rank' ? s.modeActive : ''}`} onClick={() => setMode('rank')}>Rank All 13</button>
      </div>

      <div className={s.body}>
        {/* Left: Strategy selector + params */}
        {mode !== 'rank' && (
          <div className={s.leftPanel}>
            {/* Strategy selector */}
            <div className={s.stratSelector}>
              <div className={s.fieldLabel}>Strategy</div>
              {strategies.map(st => (
                <button key={st.id} className={`${s.stratBtn} ${selected === st.id ? s.stratActive : ''}`}
                  onClick={() => selectStrategy(st.id)}>
                  {st.id.replace(/_/g, ' ')}
                  <span className={s.paramCount}>{Object.keys(st.params).length}p</span>
                </button>
              ))}
            </div>

            {/* Global params */}
            <div className={s.globalParams}>
              <div className={s.fieldLabel}>Backtest Config</div>
              <div className={s.fieldRow}><label>Start</label><input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} /></div>
              <div className={s.fieldRow}><label>End</label><input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} /></div>
              <div className={s.fieldRow}><label>Capital</label><input type="number" value={capital} onChange={e => setCapital(+e.target.value)} /></div>
              <div className={s.fieldRow}><label>Max Sym/Day</label><input type="number" value={maxSymbols} onChange={e => setMaxSymbols(+e.target.value)} /></div>
              <div className={s.fieldRow}><label>Max Positions</label><input type="number" value={maxPositions} onChange={e => setMaxPositions(+e.target.value)} /></div>
              <div className={s.fieldRow}><label>Risk %</label><input type="number" step="0.001" value={riskPct} onChange={e => setRiskPct(+e.target.value)} /></div>
            </div>
          </div>
        )}

        {/* Center: Strategy parameters */}
        <div className={s.centerPanel}>
          {mode === 'rank' ? (
            <div className={s.rankInfo}>
              <div className={s.fieldLabel}>Backtest Config</div>
              <div className={s.rankFields}>
                <div className={s.fieldRow}><label>Start</label><input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} /></div>
                <div className={s.fieldRow}><label>End</label><input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} /></div>
                <div className={s.fieldRow}><label>Capital</label><input type="number" value={capital} onChange={e => setCapital(+e.target.value)} /></div>
                <div className={s.fieldRow}><label>Max Sym/Day</label><input type="number" value={maxSymbols} onChange={e => setMaxSymbols(+e.target.value)} /></div>
              </div>
              <button className={s.runBtn} onClick={runRankAll} disabled={running}>
                {running ? `${progress.pct.toFixed(0)}% — ${progress.msg}` : '▶ Rank All 13 Strategies (parallel)'}
              </button>
            </div>
          ) : selected ? (
            <>
              <div className={s.paramHeader}>
                <span className={s.fieldLabel}>{selected.replace(/_/g, ' ').toUpperCase()} — Parameters</span>
                {mode === 'sweep' && <span className={s.comboCount}>{combos} combinations</span>}
              </div>
              <div className={s.paramGrid}>
                {Object.entries(params).map(([key, val]) => (
                  <div key={key} className={s.paramRow}>
                    <div className={s.paramLeft}>
                      {mode === 'sweep' && (
                        <button className={`${s.sweepToggle} ${sweepEnabled.has(key) ? s.sweepOn : ''}`}
                          onClick={() => toggleSweep(key)} title="Include in sweep">⟳</button>
                      )}
                      <label className={s.paramLabel}>{key.replace(/_/g, ' ')}</label>
                    </div>
                    <div className={s.paramRight}>
                      {typeof val === 'boolean' ? (
                        <select value={String(val)} onChange={e => updateParam(key, e.target.value)}>
                          <option value="true">true</option><option value="false">false</option>
                        </select>
                      ) : (
                        <input type={typeof val === 'number' ? 'number' : 'text'} step="any"
                          value={String(val)} onChange={e => updateParam(key, e.target.value)} />
                      )}
                      {mode === 'sweep' && sweepEnabled.has(key) && (
                        <input className={s.sweepInput} placeholder="val1, val2, val3..."
                          value={sweepGrid[key] ?? ''} onChange={e => setSweepGrid(g => ({ ...g, [key]: e.target.value }))} />
                      )}
                    </div>
                  </div>
                ))}
              </div>
              <div className={s.actionBar}>
                {mode === 'single' && <button className={s.runBtn} onClick={runSingle} disabled={running}>{running ? `${progress.pct.toFixed(0)}%` : '▶ Run Backtest'}</button>}
                {mode === 'sweep' && <button className={s.runBtn} onClick={runSweep} disabled={running || combos < 2}>{running ? `${progress.pct.toFixed(0)}% — ${progress.msg}` : `▶ Run Sweep (${combos} combos)`}</button>}
              </div>
            </>
          ) : <div className={s.emptyCenter}>Select a strategy from the left panel</div>}

          {/* Progress */}
          {running && <div className={s.progressBar}><div className={s.progressFill} style={{ width: `${progress.pct}%` }} /><span className={s.progressText}>{progress.msg}</span></div>}

          {/* Single Result */}
          {result && !sweepResults && (
            <div className={s.resultSection}>
              <div className={s.fieldLabel}>Results — {result.strategy_name || result.strategy_id}</div>
              <div className={s.metricsGrid}>
                {[
                  ['Trades', result.total_trades], ['Win Rate', `${(result.win_rate*100).toFixed(1)}%`],
                  ['Net P&L', `$${result.net_pnl?.toLocaleString()}`], ['PF', result.profit_factor?.toFixed(2)],
                  ['Sharpe', result.sharpe?.toFixed(2)], ['Max DD', `${result.max_drawdown_pct?.toFixed(1)}%`],
                  ['Avg Win', `$${result.avg_win?.toFixed(0)}`], ['Avg Loss', `$${result.avg_loss?.toFixed(0)}`],
                  ['Avg Hold', `${result.avg_hold_min ?? 0}m`], ['Status', ''],
                ].map(([label, val], i) => (
                  <div key={i} className={s.metricItem}>
                    <span className={s.metricLabel}>{String(label)}</span>
                    {label === 'Status' ? <StatusBadge pf={result.profit_factor} sharpe={result.sharpe} /> : <span className={s.metricValue}>{String(val)}</span>}
                  </div>
                ))}
              </div>
              {result.trades && result.trades.length > 0 && (
                <details className={s.tradeDetails}>
                  <summary>{result.trades.length} trades</summary>
                  <div className={s.tradeTable}>
                    <table>
                      <thead><tr><th>Time</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead>
                      <tbody>{result.trades.slice(0, 50).map((t, i) => (
                        <tr key={i} className={Number(t.pnl ?? 0) >= 0 ? s.rowUp : s.rowDown}>
                          <td className={s.mono}>{String(t.entry_time ?? '')}</td>
                          <td className={s.ticker}>{String(t.symbol ?? '')}</td>
                          <td className={String(t.direction) === 'long' ? s.up : s.down}>{String(t.direction ?? '').toUpperCase()}</td>
                          <td className={s.mono}>${Number(t.entry_price ?? 0).toFixed(2)}</td>
                          <td className={s.mono}>${Number(t.exit_price ?? 0).toFixed(2)}</td>
                          <td className={`${s.mono} ${Number(t.pnl ?? 0) >= 0 ? s.up : s.down}`}>${Number(t.pnl ?? 0).toFixed(2)}</td>
                          <td>{String(t.exit_reason ?? '')}</td>
                        </tr>
                      ))}</tbody>
                    </table>
                  </div>
                </details>
              )}
            </div>
          )}

          {/* Sweep Results */}
          {sweepResults && (
            <div className={s.resultSection}>
              <div className={s.fieldLabel}>Sweep Results — {sweepResults.length} combinations</div>
              <div className={s.sweepTable}>
                <table>
                  <thead><tr><th>#</th><th>Params</th><th>Trades</th><th>WR</th><th>PF</th><th>Sharpe</th><th>P&L</th><th>DD</th></tr></thead>
                  <tbody>{sweepResults.slice(0, 50).map((r: Record<string, unknown>, i: number) => (
                    <tr key={i} className={Number(r.net_pnl ?? 0) >= 0 ? s.rowUp : s.rowDown}
                      onClick={() => { if (r.params) setParams(r.params as Record<string, unknown>) }}>
                      <td>{i + 1}</td>
                      <td className={s.mono}>{Object.entries(r.params as Record<string, unknown> ?? {}).map(([k,v]) => `${k}=${v}`).join(', ')}</td>
                      <td className={s.mono}>{String(r.total_trades ?? 0)}</td>
                      <td className={s.mono}>{(Number(r.win_rate ?? 0) * 100).toFixed(1)}%</td>
                      <td className={`${s.mono} ${Number(r.profit_factor ?? 0) >= 1.5 ? s.up : ''}`}>{Number(r.profit_factor ?? 0).toFixed(2)}</td>
                      <td className={`${s.mono} ${Number(r.sharpe ?? 0) >= 2 ? s.up : ''}`}>{Number(r.sharpe ?? 0).toFixed(2)}</td>
                      <td className={`${s.mono} ${Number(r.net_pnl ?? 0) >= 0 ? s.up : s.down}`}>${Number(r.net_pnl ?? 0).toLocaleString()}</td>
                      <td className={s.mono}>{Number(r.max_drawdown_pct ?? 0).toFixed(1)}%</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}

          {/* Rankings */}
          {rankings.length > 0 && (
            <div className={s.resultSection}>
              <div className={s.fieldLabel}>All 13 Strategies Ranked</div>
              <table className={s.rankTable}>
                <thead><tr><th>#</th><th>Strategy</th><th>Trades</th><th>WR</th><th>PF</th><th>Sharpe</th><th>P&L</th><th>DD</th><th>Status</th></tr></thead>
                <tbody>{rankings.map((r, i) => (
                  <tr key={r.strategy_id} className={r.net_pnl >= 0 ? s.rowUp : s.rowDown}>
                    <td>{i + 1}</td>
                    <td className={s.ticker}>{r.strategy_name || r.strategy_id}</td>
                    <td className={s.mono}>{r.total_trades}</td>
                    <td className={s.mono}>{(r.win_rate*100).toFixed(1)}%</td>
                    <td className={`${s.mono} ${r.profit_factor >= 1.5 ? s.up : ''}`}>{r.profit_factor.toFixed(2)}</td>
                    <td className={`${s.mono} ${r.sharpe >= 2 ? s.up : ''}`}>{r.sharpe.toFixed(2)}</td>
                    <td className={`${s.mono} ${r.net_pnl >= 0 ? s.up : s.down}`}>${r.net_pnl.toLocaleString()}</td>
                    <td className={s.mono}>{r.max_drawdown_pct.toFixed(1)}%</td>
                    <td><StatusBadge pf={r.profit_factor} sharpe={r.sharpe} /></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
