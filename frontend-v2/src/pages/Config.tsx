import { useState, useEffect, useMemo, useCallback } from 'react'
import { useToast } from '../components/Toast'
import { setApiKey } from '../lib/api'
import s from './Config.module.css'

// Config sections matching the embedded UI
const SECTIONS: Record<string, { label: string; keys: string[] }> = {
  gap: { label: 'Gap Detection', keys: ['gap_threshold', 'max_gap_pct', 'vol_ratio_max', 'min_avg_volume', 'min_price', 'min_gap_fill_pct'] },
  sizing: { label: 'Position Sizing', keys: ['max_positions', 'initial_capital', 'risk_pct', 'kelly_fraction', 'max_notional'] },
  risk: { label: 'Risk Management', keys: ['stop_pct', 'daily_loss_limit', 'max_consec_losses', 'max_drawdown', 'thin_day_threshold', 'min_hold_minutes'] },
  timing: { label: 'Entry & Exit Timing', keys: ['entry_cutoff_hour', 'entry_cutoff_min', 'time_exit_hour', 'time_exit_min', 'eod_exit_hour', 'eod_exit_min', 'bounce_entry_pct', 'partial_target_pct'] },
  execution: { label: 'Order Execution', keys: ['limit_orders_only', 'slippage_pct', 'borrow_rate_annual', 'max_pct_adv', 'limit_offset_pct', 'adverse_fill_pct'] },
  trailing: { label: 'Trailing Stop', keys: ['trailing_stop_enabled', 'trailing_activation_pct', 'trailing_distance_pct'] },
  adaptive: { label: 'Adaptive Stops', keys: ['adaptive_stops', 'stop_gap_fraction', 'stop_min_pct', 'stop_max_pct'] },
  regime: { label: 'Market Regime Filter', keys: ['regime_filter', 'regime_spy_gap_limit', 'regime_spy_block_pct', 'regime_vix_threshold'] },
  reentry: { label: 'Re-entry After Stop', keys: ['reentry_enabled', 'reentry_cooldown_minutes', 'reentry_max_per_symbol', 'reentry_stop_pct', 'reentry_trigger_drop_pct'] },
  gapdown: { label: 'Gap-Down Fading (Longs)', keys: ['gap_downs_enabled', 'gap_down_threshold', 'gap_down_max_pct', 'gap_down_vol_ratio_max'] },
  ddbreaker: { label: 'Drawdown Circuit Breaker', keys: ['dd_circuit_breaker', 'dd_tier1_pct', 'dd_tier1_scale', 'dd_tier2_pct', 'dd_tier2_scale', 'dd_tier2_max_pos', 'dd_hard_stop'] },
  llm: { label: 'Rudra (LLM Supervisor)', keys: ['llm_enabled', 'llm_url', 'llm_model', 'llm_timeout', 'llm_max_failures', 'llm_circuit_reset', 'llm_max_hold_overrides', 'llm_provider', 'llm_api_key'] },
  catalyst: { label: 'Catalyst Detection', keys: ['catalyst_enabled', 'catalyst_skip_earnings', 'catalyst_earnings_penalty', 'catalyst_news_penalty', 'catalyst_noise_bonus'] },
  pyramid: { label: 'Pyramiding', keys: ['pyramid_enabled', 'pyramid_max_adds', 'pyramid_min_profit_pct', 'pyramid_size_decay', 'pyramid_interval_pct'] },
  sweep: { label: 'Sweep Detector', keys: ['sweep_entry_enabled', 'sweep_lookback_bars', 'sweep_max_time_bars', 'sweep_recovery_pct'] },
  intraday: { label: 'Intraday Strategies', keys: ['intraday_enabled', 'intraday_strategies', 'intraday_scan_interval', 'intraday_watchlist_size', 'intraday_max_entries', 'intraday_risk_pct', 'intraday_daily_loss_limit', 'intraday_max_position_pct'] },
  capital: { label: 'Capital Partitioning', keys: ['capital_partition', 'gap_fade_capital_pct', 'intraday_capital_pct', 'gap_fade_max_positions', 'intraday_max_positions', 'capital_overflow', 'capital_overflow_hour', 'capital_overflow_min'] },
  swing: { label: 'Swing Trading', keys: ['swing_enabled', 'swing_max_positions', 'swing_max_weekly_trades', 'swing_risk_pct'] },
  orb: { label: 'ORB Confirmation', keys: ['orb_enabled', 'orb_atr_fraction', 'orb_atr_period', 'orb_dynamic_stop', 'orb_min_pct', 'orb_max_pct'] },
  autostart: { label: 'Auto-Start', keys: ['auto_start'] },
  connection: { label: 'API Connection', keys: ['_api_key'] },
}

function humanLabel(key: string): string {
  return key
    .replace(/_/g, ' ')
    .replace(/\b(pct|Pct)\b/g, '%')
    .replace(/\b(min|Min)\b/g, 'Min')
    .replace(/\bdd\b/gi, 'DD')
    .replace(/\bllm\b/gi, 'LLM')
    .replace(/\borb\b/gi, 'ORB')
    .replace(/\bvix\b/gi, 'VIX')
    .replace(/\bspy\b/gi, 'SPY')
    .replace(/\badv\b/gi, 'ADV')
    .replace(/^./, c => c.toUpperCase())
}

function isToggle(key: string, val: unknown): boolean {
  return typeof val === 'boolean' || key.endsWith('_enabled') || key === 'dd_circuit_breaker' || key === 'adaptive_stops' || key === 'regime_filter' || key === 'capital_partition' || key === 'capital_overflow' || key === 'catalyst_skip_earnings' || key === 'limit_orders_only' || key === 'auto_start' || key === 'orb_dynamic_stop'
}

export default function Config() {
  const [config, setConfig] = useState<Record<string, unknown>>({})
  const [original, setOriginal] = useState<Record<string, unknown>>({})
  const [profiles, setProfiles] = useState<string[]>([])
  const [search, setSearch] = useState('')
  const [undoStack, setUndoStack] = useState<Record<string, unknown>[]>([])
  const [tab, setTab] = useState<string | null>(null)
  const { showToast } = useToast()

  // Load config from state
  useEffect(() => {
    fetch('/api/state').then(r => r.json()).then(d => {
      const cfg = d.config ?? {}
      setConfig(cfg)
      setOriginal(cfg)
    }).catch(() => {})
    fetch('/api/config/profiles').then(r => r.json()).then(d => {
      setProfiles(Array.isArray(d) ? d.map((p: Record<string, unknown>) => String(p.name ?? p)) : d.profiles?.map((p: Record<string, unknown>) => String(p.name ?? p)) ?? [])
    }).catch(() => {})
  }, [])

  const pendingChanges = useMemo(() => {
    return Object.keys(config).filter(k => config[k] !== original[k])
  }, [config, original])

  const updateField = useCallback((key: string, value: unknown) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }, [])

  const saveConfig = async () => {
    setUndoStack(prev => [...prev.slice(-49), { ...original }])
    try {
      // Handle special _api_key field
      if (config._api_key) {
        setApiKey(String(config._api_key))
      }
      const toSend = { ...config }
      delete toSend._api_key
      await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(toSend) })
      setOriginal({ ...config })
      showToast(`Saved ${pendingChanges.length} changes`, 'success')
    } catch (e) {
      showToast('Failed to save config', 'error')
    }
  }

  const undo = () => {
    if (undoStack.length === 0) return
    const prev = undoStack[undoStack.length - 1]
    setUndoStack(s => s.slice(0, -1))
    setConfig(prev)
    setOriginal(prev)
    fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(prev) })
    showToast('Config reverted', 'info')
  }

  const resetDefaults = async () => {
    try {
      const d = await fetch('/api/config/defaults').then(r => r.json())
      setConfig(d)
      showToast('Reset to defaults (unsaved)', 'info')
    } catch { showToast('Failed to fetch defaults', 'error') }
  }

  const loadProfile = async (name: string) => {
    try {
      await fetch(`/api/config/profiles/${name}/load`, { method: 'POST' })
      const d = await fetch('/api/state').then(r => r.json())
      setConfig(d.config ?? {})
      setOriginal(d.config ?? {})
      showToast(`Loaded profile: ${name}`, 'success')
    } catch { showToast('Failed to load profile', 'error') }
  }

  const testLlm = async () => {
    try {
      const r = await fetch('/api/llm/test', { method: 'POST' }).then(r => r.json())
      showToast(r.status === 'ok' ? `LLM connected: ${r.model ?? 'ok'}` : `LLM error: ${r.error ?? 'failed'}`, r.status === 'ok' ? 'success' : 'error')
    } catch { showToast('LLM test failed', 'error') }
  }

  // Filter sections by search
  const filteredSections = useMemo(() => {
    const q = search.toLowerCase()
    if (!q) return Object.entries(SECTIONS)
    return Object.entries(SECTIONS).filter(([, sec]) =>
      sec.label.toLowerCase().includes(q) || sec.keys.some(k => k.includes(q))
    ).map(([id, sec]) => [id, { ...sec, keys: sec.keys.filter(k => k.includes(q) || sec.label.toLowerCase().includes(q)) }] as [string, typeof sec])
  }, [search])

  const activeTab = tab ?? filteredSections[0]?.[0] ?? 'gap'

  return (
    <div className={s.page}>
      {/* Top bar: search + profiles + actions */}
      <div className={s.topBar}>
        <input className={s.search} placeholder="Search parameters... (Ctrl+K)" value={search}
          onChange={e => setSearch(e.target.value)} />
        <div className={s.profileBar}>
          {profiles.length > 0 && (
            <select className={s.select} onChange={e => e.target.value && loadProfile(e.target.value)} defaultValue="">
              <option value="" disabled>Profiles</option>
              {profiles.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          )}
          <button className={s.btn} onClick={resetDefaults}>Reset Defaults</button>
          {undoStack.length > 0 && <button className={s.btn} onClick={undo}>Undo</button>}
        </div>
      </div>

      <div className={s.layout}>
        {/* Left: section tabs */}
        <nav className={s.tabs}>
          {filteredSections.map(([id, sec]) => (
            <button key={id} className={`${s.tabItem} ${activeTab === id ? s.tabActive : ''}`}
              onClick={() => setTab(id)}>
              {sec.label}
              {sec.keys.some(k => pendingChanges.includes(k)) && <span className={s.tabDot} />}
            </button>
          ))}
        </nav>

        {/* Right: config fields */}
        <div className={s.fields}>
          {filteredSections.filter(([id]) => id === activeTab).map(([id, sec]) => (
            <div key={id}>
              <div className={s.sectionTitle}>{sec.label}</div>
              <div className={s.fieldGrid}>
                {sec.keys.map(key => {
                  // Special: API key field
                  if (key === '_api_key') {
                    return (
                      <div key={key} className={s.field}>
                        <label className={s.fieldLabel}>API Key (frontend)</label>
                        <input className={s.input} type="password" placeholder="GAP_FADE_API_KEY"
                          value={String(config._api_key ?? localStorage.getItem('rudra_api_key') ?? '')}
                          onChange={e => updateField('_api_key', e.target.value)} />
                      </div>
                    )
                  }
                  const val = config[key]
                  if (val === undefined) return null
                  const changed = val !== original[key]
                  if (isToggle(key, val)) {
                    return (
                      <div key={key} className={`${s.field} ${s.fieldToggle}`}>
                        <label className={`${s.fieldLabel} ${changed ? s.changed : ''}`}>{humanLabel(key)}</label>
                        <button className={`${s.toggle} ${val ? s.toggleOn : ''}`}
                          onClick={() => updateField(key, !val)}>
                          <span className={s.toggleThumb} />
                        </button>
                      </div>
                    )
                  }
                  if (typeof val === 'number') {
                    return (
                      <div key={key} className={s.field}>
                        <label className={`${s.fieldLabel} ${changed ? s.changed : ''}`}>{humanLabel(key)}</label>
                        <input className={s.input} type="number" step="any" value={val}
                          onChange={e => updateField(key, Number(e.target.value))} />
                      </div>
                    )
                  }
                  return (
                    <div key={key} className={s.field}>
                      <label className={`${s.fieldLabel} ${changed ? s.changed : ''}`}>{key}</label>
                      <input className={s.input} type="text" value={String(val ?? '')}
                        onChange={e => updateField(key, e.target.value)} />
                    </div>
                  )
                })}
                {id === 'llm' && (
                  <div className={s.field}>
                    <label className={s.fieldLabel}>&nbsp;</label>
                    <button className={s.btnAccent} onClick={testLlm}>Test LLM Connection</button>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Sticky save footer */}
      {pendingChanges.length > 0 && (
        <div className={s.footer}>
          <span className={s.footerDot} />
          <span>{pendingChanges.length} changes pending</span>
          <div className={s.footerActions}>
            <button className={s.btn} onClick={() => setConfig({ ...original })}>Discard</button>
            <button className={s.btnAccent} onClick={saveConfig}>Save & Apply</button>
          </div>
        </div>
      )}
    </div>
  )
}
