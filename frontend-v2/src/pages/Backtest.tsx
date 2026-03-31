import { useState, useEffect, useRef } from 'react'
import { useToast } from '../components/Toast'
import s from './Backtest.module.css'

interface BtForm {
  symbol: string; universe: string; strategy: string
  start_date: string; end_date: string
  gap_threshold: number; max_gap_pct: number; vol_ratio_max: number
  stop_pct: number; max_positions: number; slippage_pct: number; cover_frac: number
  use_1min: boolean
  adaptive_stops: boolean; stop_gap_fraction: number; stop_min_pct: number; stop_max_pct: number
  regime_filter: boolean; regime_spy_gap_limit: number; regime_spy_block_pct: number; regime_vix_threshold: number
  reentry_enabled: boolean; reentry_cooldown_minutes: number; reentry_max_per_symbol: number; reentry_stop_pct: number; reentry_trigger_drop_pct: number
  gap_downs_enabled: boolean; gap_down_threshold: number; gap_down_max_pct: number
  orb_enabled: boolean; orb_atr_fraction: number
  exclude_leveraged: boolean
  dd_circuit_breaker: boolean; dd_tier1_pct: number; dd_tier1_scale: number; dd_tier2_pct: number; dd_tier2_scale: number; dd_hard_stop: number
}

const DEFAULT: BtForm = {
  symbol: '', universe: 'study', strategy: '',
  start_date: '2024-01-01', end_date: '2025-06-30',
  gap_threshold: 0.03, max_gap_pct: 0.15, vol_ratio_max: 8,
  stop_pct: 0.03, max_positions: 5, slippage_pct: 0.001, cover_frac: 0.5,
  use_1min: false,
  adaptive_stops: false, stop_gap_fraction: 0.4, stop_min_pct: 0.015, stop_max_pct: 0.05,
  regime_filter: false, regime_spy_gap_limit: 0.015, regime_spy_block_pct: 0.02, regime_vix_threshold: 30,
  reentry_enabled: false, reentry_cooldown_minutes: 30, reentry_max_per_symbol: 1, reentry_stop_pct: 0.025, reentry_trigger_drop_pct: 0.01,
  gap_downs_enabled: false, gap_down_threshold: 0.04, gap_down_max_pct: 0.12,
  orb_enabled: false, orb_atr_fraction: 0.5,
  exclude_leveraged: true,
  dd_circuit_breaker: false, dd_tier1_pct: 5, dd_tier1_scale: 0.5, dd_tier2_pct: 10, dd_tier2_scale: 0.25, dd_hard_stop: 15,
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className={s.toggle}>
      <button className={`${s.toggleBtn} ${checked ? s.toggleOn : ''}`} onClick={() => onChange(!checked)}>
        <span className={s.toggleThumb} />
      </button>
      <span className={s.toggleLabel}>{label}</span>
    </div>
  )
}

function Field({ label, value, onChange, type = 'number', step }: { label: string; value: number | string; onChange: (v: string) => void; type?: string; step?: string }) {
  return (
    <div className={s.field}>
      <label>{label}</label>
      <input type={type} step={step ?? 'any'} value={value} onChange={e => onChange(e.target.value)} />
    </div>
  )
}

export default function Backtest() {
  const [form, setForm] = useState<BtForm>(DEFAULT)
  const [strategies, setStrategies] = useState<{ id: string; name: string }[]>([])
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState<{ msg: string; pct: number }>({ msg: '', pct: 0 })
  const [results, setResults] = useState<Record<string, unknown> | null>(null)
  const [log, setLog] = useState<string[]>([])
  const logRef = useRef<HTMLDivElement>(null)
  const { showToast } = useToast()

  useEffect(() => {
    fetch('/api/strategies').then(r => r.json()).then(d => {
      const list = d?.strategies
      setStrategies(Array.isArray(list) ? list : [])
    }).catch(() => {})
  }, [])

  const set = <K extends keyof BtForm>(key: K, val: BtForm[K]) => setForm(f => ({ ...f, [key]: val }))

  const run = async () => {
    setRunning(true); setResults(null); setLog([])
    try {
      await fetch('/api/backtest', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) })
      const poll = setInterval(async () => {
        const st = await fetch('/api/backtest/status').then(r => r.json())
        setProgress({ msg: st.message ?? '', pct: st.progress ?? 0 })
        if (st.status === 'complete' || st.status === 'idle') {
          clearInterval(poll); setRunning(false)
          const res = await fetch('/api/backtest/results').then(r => r.json())
          setResults(res); showToast('Backtest complete', 'success')
        }
        if (st.status === 'error') { clearInterval(poll); setRunning(false); showToast('Backtest failed', 'error') }
      }, 2000)
    } catch { setRunning(false); showToast('Failed to start backtest', 'error') }
  }

  const cancel = () => { fetch('/api/backtest/cancel', { method: 'POST' }); setRunning(false) }

  const m = results as Record<string, unknown> | null

  return (
    <div className={s.page}>
      {/* Controls */}
      <div className={s.controlsGrid}>
        <Field label="Symbol(s)" value={form.symbol} onChange={v => set('symbol', v)} type="text" />
        <div className={s.field}><label>Universe</label>
          <select value={form.universe} onChange={e => set('universe', e.target.value)}>
            <option value="study">Study (167)</option><option value="alpaca">All Tradeable</option>
          </select></div>
        <div className={s.field}><label>Strategy</label>
          <select value={form.strategy} onChange={e => set('strategy', e.target.value)}>
            <option value="">Engine Default</option>
            {strategies.map(st => <option key={st.id} value={st.id}>{st.name}</option>)}
          </select></div>
        <Field label="Start Date" value={form.start_date} onChange={v => set('start_date', v)} type="date" />
        <Field label="End Date" value={form.end_date} onChange={v => set('end_date', v)} type="date" />
        <Field label="Gap % Min" value={form.gap_threshold} onChange={v => set('gap_threshold', +v)} />
        <Field label="Max Gap %" value={form.max_gap_pct} onChange={v => set('max_gap_pct', +v)} />
        <Field label="Vol Max" value={form.vol_ratio_max} onChange={v => set('vol_ratio_max', +v)} />
        <Field label="Stop %" value={form.stop_pct} onChange={v => set('stop_pct', +v)} />
        <Field label="Max Pos" value={form.max_positions} onChange={v => set('max_positions', +v)} />
        <Field label="Slip %" value={form.slippage_pct} onChange={v => set('slippage_pct', +v)} />
        <Field label="Cover Frac" value={form.cover_frac} onChange={v => set('cover_frac', +v)} />
      </div>

      {/* Feature Toggles */}
      <div className={s.togglesSection}>
        <Toggle label="1-Min Bars (slow)" checked={form.use_1min} onChange={v => set('use_1min', v)} />
        <Toggle label="Exclude Leveraged ETFs" checked={form.exclude_leveraged} onChange={v => set('exclude_leveraged', v)} />

        <Toggle label="Adaptive Stops" checked={form.adaptive_stops} onChange={v => set('adaptive_stops', v)} />
        {form.adaptive_stops && <div className={s.subFields}>
          <Field label="Gap Frac" value={form.stop_gap_fraction} onChange={v => set('stop_gap_fraction', +v)} />
          <Field label="Min %" value={form.stop_min_pct} onChange={v => set('stop_min_pct', +v)} />
          <Field label="Max %" value={form.stop_max_pct} onChange={v => set('stop_max_pct', +v)} />
        </div>}

        <Toggle label="Market Regime Filter" checked={form.regime_filter} onChange={v => set('regime_filter', v)} />
        {form.regime_filter && <div className={s.subFields}>
          <Field label="SPY Gap Limit" value={form.regime_spy_gap_limit} onChange={v => set('regime_spy_gap_limit', +v)} />
          <Field label="SPY Block %" value={form.regime_spy_block_pct} onChange={v => set('regime_spy_block_pct', +v)} />
          <Field label="VIX Threshold" value={form.regime_vix_threshold} onChange={v => set('regime_vix_threshold', +v)} />
        </div>}

        <Toggle label="Re-entry After Stop" checked={form.reentry_enabled} onChange={v => set('reentry_enabled', v)} />
        {form.reentry_enabled && <div className={s.subFields}>
          <Field label="Cooldown Min" value={form.reentry_cooldown_minutes} onChange={v => set('reentry_cooldown_minutes', +v)} />
          <Field label="Max/Symbol" value={form.reentry_max_per_symbol} onChange={v => set('reentry_max_per_symbol', +v)} />
          <Field label="Stop %" value={form.reentry_stop_pct} onChange={v => set('reentry_stop_pct', +v)} />
          <Field label="Trigger Drop %" value={form.reentry_trigger_drop_pct} onChange={v => set('reentry_trigger_drop_pct', +v)} />
        </div>}

        <Toggle label="Gap-Down Fading (Longs)" checked={form.gap_downs_enabled} onChange={v => set('gap_downs_enabled', v)} />
        {form.gap_downs_enabled && <div className={s.subFields}>
          <Field label="Gap Down %" value={form.gap_down_threshold} onChange={v => set('gap_down_threshold', +v)} />
          <Field label="Max Gap Down %" value={form.gap_down_max_pct} onChange={v => set('gap_down_max_pct', +v)} />
        </div>}

        <Toggle label="ORB Confirmation" checked={form.orb_enabled} onChange={v => set('orb_enabled', v)} />
        {form.orb_enabled && <div className={s.subFields}>
          <Field label="ATR Fraction" value={form.orb_atr_fraction} onChange={v => set('orb_atr_fraction', +v)} />
        </div>}

        <Toggle label="Drawdown Circuit Breaker" checked={form.dd_circuit_breaker} onChange={v => set('dd_circuit_breaker', v)} />
        {form.dd_circuit_breaker && <div className={s.subFields}>
          <Field label="Tier 1 DD%" value={form.dd_tier1_pct} onChange={v => set('dd_tier1_pct', +v)} />
          <Field label="Tier 1 Scale" value={form.dd_tier1_scale} onChange={v => set('dd_tier1_scale', +v)} />
          <Field label="Tier 2 DD%" value={form.dd_tier2_pct} onChange={v => set('dd_tier2_pct', +v)} />
          <Field label="Tier 2 Scale" value={form.dd_tier2_scale} onChange={v => set('dd_tier2_scale', +v)} />
          <Field label="Hard Stop %" value={form.dd_hard_stop} onChange={v => set('dd_hard_stop', +v)} />
        </div>}
      </div>

      {/* Action buttons */}
      <div className={s.actions}>
        <button className={s.btnRun} onClick={run} disabled={running}>{running ? 'Running...' : '▶ Run Backtest'}</button>
        {running && <button className={s.btnCancel} onClick={cancel}>Cancel</button>}
      </div>

      {/* Progress */}
      {running && (
        <div className={s.progressBar}>
          <div className={s.progressFill} style={{ width: `${progress.pct}%` }} />
          <span className={s.progressText}>{progress.msg || `${progress.pct}%`}</span>
        </div>
      )}

      {/* Results */}
      {m && (
        <div className={s.results}>
          <div className={s.resultsTitle}>Results</div>
          <div className={s.metricsGrid}>
            {[
              ['Final Equity', `$${Number(m.final_equity ?? m.ending_equity ?? 0).toLocaleString()}`],
              ['Net P&L', `$${Number(m.net_pnl ?? m.total_pnl ?? 0).toFixed(2)}`],
              ['Return', `${Number(m.return_pct ?? 0).toFixed(1)}%`],
              ['Total Trades', String(m.total_trades ?? 0)],
              ['Win Rate', `${(Number(m.win_rate ?? 0) * 100).toFixed(1)}%`],
              ['Profit Factor', Number(m.profit_factor ?? 0).toFixed(2)],
              ['Sharpe', Number(m.sharpe ?? 0).toFixed(2)],
              ['Max DD', `${Number(m.max_drawdown_pct ?? 0).toFixed(1)}%`],
              ['Avg Win', `$${Number(m.avg_win ?? 0).toFixed(2)}`],
              ['Avg Loss', `$${Number(m.avg_loss ?? 0).toFixed(2)}`],
              ['Avg Hold', `${Number(m.avg_holding_min ?? 0).toFixed(0)} min`],
              ['Best Trade', `$${Number(m.best_trade ?? 0).toFixed(2)}`],
            ].map(([label, value]) => (
              <div key={label} className={s.metricItem}>
                <span className={s.metricLabel}>{label}</span>
                <span className={s.metricValue}>{value}</span>
              </div>
            ))}
          </div>

          {/* Trades table */}
          {Array.isArray(m.trades) && m.trades.length > 0 && (
            <div className={s.tradesSection}>
              <div className={s.resultsTitle}>Trades ({(m.trades as unknown[]).length})</div>
              <div className={s.tableWrap}>
                <table className={s.table}>
                  <thead><tr><th>Date</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Exit</th><th>P&L</th><th>P&L%</th><th>Hold</th><th>Reason</th></tr></thead>
                  <tbody>{(m.trades as Record<string, unknown>[]).map((t, i) => (
                    <tr key={i} className={Number(t.pnl ?? 0) >= 0 ? s.rowUp : s.rowDown}>
                      <td className={s.mono}>{String(t.entry_time ?? t.date ?? '')}</td>
                      <td className={s.ticker}>{String(t.symbol ?? '')}</td>
                      <td className={String(t.direction ?? t.side ?? '') === 'long' ? s.up : s.down}>{String(t.direction ?? t.side ?? '').toUpperCase()}</td>
                      <td className={s.mono}>${Number(t.entry_price ?? 0).toFixed(2)}</td>
                      <td className={s.mono}>${Number(t.exit_price ?? 0).toFixed(2)}</td>
                      <td className={`${s.mono} ${Number(t.pnl ?? 0) >= 0 ? s.up : s.down}`}>{Number(t.pnl ?? 0) >= 0 ? '+' : ''}${Number(t.pnl ?? 0).toFixed(2)}</td>
                      <td className={`${s.mono} ${Number(t.pnl_pct ?? 0) >= 0 ? s.up : s.down}`}>{(Number(t.pnl_pct ?? 0) * 100).toFixed(2)}%</td>
                      <td className={s.mono}>{String(t.holding_minutes ?? t.hold ?? '—')}m</td>
                      <td>{String(t.exit_reason ?? '')}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {!running && !results && <div className={s.empty}>Configure parameters and run a backtest</div>}
    </div>
  )
}
