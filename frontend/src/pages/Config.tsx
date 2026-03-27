import { useState, useEffect, useCallback } from 'react'
import StatusBanner from '../components/StatusBanner'
import { usePolling } from '../hooks/usePolling'
import { gfApi, type GfConfigSchema } from '../api/client'

interface ToggleSwitchProps {
  enabled: boolean
  onChange: (enabled: boolean) => void
}

function ToggleSwitch({ enabled, onChange }: ToggleSwitchProps) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className={`w-10 h-5 rounded-full transition-colors relative ${enabled ? 'bg-green' : 'bg-bg-card-hover'}`}
    >
      <span
        className={`absolute top-0.5 w-4 h-4 rounded-full bg-white transition-transform ${enabled ? 'translate-x-5' : 'translate-x-0.5'}`}
      />
    </button>
  )
}

export default function Config() {
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({})
  const [dirtyKeys, setDirtyKeys] = useState<Set<string>>(new Set())
  const [saving, setSaving] = useState(false)
  const [saveMsg, setSaveMsg] = useState<string | null>(null)

  // Fetch live config
  const { data: liveConfig, error: configError, refresh: refreshConfig } = usePolling<Record<string, unknown>>(
    useCallback(() => gfApi.getConfig(), []),
    30000,
  )

  // Fetch config schema (for field descriptions/types)
  const { data: schema } = usePolling<GfConfigSchema>(
    useCallback(() => gfApi.getConfigSchema(), []),
    60000,
  )

  // Fetch config change history
  const { data: history } = usePolling<Record<string, unknown>[]>(
    useCallback(() => gfApi.getConfigHistory(), []),
    30000,
  )

  // Sync live config to local state (only for non-dirty fields)
  useEffect(() => {
    if (liveConfig) {
      setConfigValues(prev => {
        const next = { ...liveConfig }
        // Preserve dirty (user-edited) values
        for (const key of dirtyKeys) {
          if (key in prev) {
            next[key] = prev[key]
          }
        }
        return next
      })
    }
  }, [liveConfig, dirtyKeys])

  const handleChange = (key: string, value: unknown) => {
    setConfigValues(prev => ({ ...prev, [key]: value }))
    setDirtyKeys(prev => new Set(prev).add(key))
  }

  const handleSave = async () => {
    if (dirtyKeys.size === 0) return
    setSaving(true)
    setSaveMsg(null)
    try {
      const payload: Record<string, unknown> = {}
      for (const key of dirtyKeys) {
        payload[key] = configValues[key]
      }
      await gfApi.updateConfig(payload)
      setDirtyKeys(new Set())
      setSaveMsg('Configuration saved successfully')
      refreshConfig()
      setTimeout(() => setSaveMsg(null), 3000)
    } catch (err) {
      setSaveMsg(`Save failed: ${String(err)}`)
    } finally {
      setSaving(false)
    }
  }

  const handleDiscard = () => {
    if (liveConfig) {
      setConfigValues({ ...liveConfig })
    }
    setDirtyKeys(new Set())
  }

  // Categorize config fields
  const configEntries = Object.entries(configValues).filter(
    ([key]) => !key.startsWith('_') && key !== 'class'
  )

  const riskFields = configEntries.filter(([key]) =>
    key.includes('max_') || key.includes('risk') || key.includes('drawdown') || key.includes('loss') || key.includes('stop')
  )
  const tradingFields = configEntries.filter(([key]) =>
    !riskFields.some(([rk]) => rk === key) &&
    (key.includes('gap') || key.includes('target') || key.includes('entry') || key.includes('exit') || key.includes('size') || key.includes('qty'))
  )
  const otherFields = configEntries.filter(([key]) =>
    !riskFields.some(([rk]) => rk === key) && !tradingFields.some(([tk]) => tk === key)
  )

  const renderField = (key: string, value: unknown) => {
    const schemaEntry = schema?.[key]
    const desc = schemaEntry?.description ?? ''
    const fieldType = schemaEntry?.type ?? typeof value

    if (typeof value === 'boolean') {
      return (
        <div key={key} className="flex items-center justify-between py-2">
          <div>
            <span className="text-sm text-text-secondary">{key}</span>
            {desc && <p className="text-[10px] text-text-muted">{desc}</p>}
          </div>
          <ToggleSwitch enabled={value} onChange={(v) => handleChange(key, v)} />
        </div>
      )
    }

    if (typeof value === 'number' || fieldType === 'float' || fieldType === 'int') {
      return (
        <div key={key} className="py-2">
          <label className="text-[10px] text-text-muted uppercase mb-1 block">{key}</label>
          <input
            type="number"
            step={fieldType === 'float' || String(value).includes('.') ? '0.01' : '1'}
            min={schemaEntry?.min}
            max={schemaEntry?.max}
            value={String(value)}
            onChange={(e) => handleChange(key, parseFloat(e.target.value) || 0)}
            className={`w-full bg-bg-primary border rounded px-3 py-2 font-mono text-sm text-text-primary focus:outline-none focus:border-cyan ${
              dirtyKeys.has(key) ? 'border-yellow' : 'border-border'
            }`}
          />
          {desc && <p className="text-[10px] text-text-muted mt-1">{desc}</p>}
          {schemaEntry?.min != null && (
            <p className="text-[10px] text-text-muted">Range: {schemaEntry.min} - {schemaEntry.max ?? 'unlimited'}</p>
          )}
        </div>
      )
    }

    if (typeof value === 'string') {
      return (
        <div key={key} className="py-2">
          <label className="text-[10px] text-text-muted uppercase mb-1 block">{key}</label>
          <input
            type="text"
            value={value}
            onChange={(e) => handleChange(key, e.target.value)}
            className={`w-full bg-bg-primary border rounded px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-cyan ${
              dirtyKeys.has(key) ? 'border-yellow' : 'border-border'
            }`}
          />
          {desc && <p className="text-[10px] text-text-muted mt-1">{desc}</p>}
        </div>
      )
    }

    // For complex types, show read-only JSON
    return (
      <div key={key} className="py-2">
        <label className="text-[10px] text-text-muted uppercase mb-1 block">{key}</label>
        <pre className="bg-bg-primary border border-border rounded px-3 py-2 text-xs text-text-secondary overflow-x-auto">
          {JSON.stringify(value, null, 2)}
        </pre>
      </div>
    )
  }

  const renderFieldGroup = (title: string, icon: string, color: string, fields: [string, unknown][]) => {
    if (fields.length === 0) return null
    return (
      <div className="bg-bg-card rounded-lg border border-border p-5">
        <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider mb-4 flex items-center gap-2">
          <span className={color}>&#9670;</span>
          {title}
        </h3>
        <div className="space-y-1">
          {fields.map(([key, value]) => renderField(key, value))}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <StatusBanner label="Gap Fade Engine" error={configError} wsConnected={true} onRetry={refreshConfig} />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-text-primary">SYSTEM CONFIGURATION</h1>
          <p className="text-xs text-text-muted mt-1">
            Live configuration from Gap Fade Engine.
            {dirtyKeys.size > 0 && (
              <span className="text-yellow ml-2">{dirtyKeys.size} unsaved change(s)</span>
            )}
          </p>
        </div>
        <div className="flex gap-2 items-center">
          {saveMsg && (
            <span className={`text-xs ${saveMsg.includes('failed') ? 'text-red' : 'text-green'}`}>
              {saveMsg}
            </span>
          )}
          <button
            onClick={handleDiscard}
            disabled={dirtyKeys.size === 0}
            className="px-4 py-2 text-xs font-semibold rounded-lg border border-border text-text-muted hover:text-text-secondary transition-colors disabled:opacity-30"
          >
            DISCARD
          </button>
          <button
            onClick={handleSave}
            disabled={dirtyKeys.size === 0 || saving}
            className="px-4 py-2 text-xs font-semibold rounded-lg bg-green text-black hover:bg-green/90 transition-colors disabled:opacity-30"
          >
            {saving ? 'SAVING...' : 'COMMIT CHANGES'}
          </button>
        </div>
      </div>

      {configEntries.length === 0 && !configError && (
        <div className="bg-bg-card rounded-lg border border-border p-8 text-center text-xs text-text-muted">
          Loading configuration...
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {renderFieldGroup('Risk Management', 'risk', 'text-yellow', riskFields)}
        {renderFieldGroup('Trading Parameters', 'trading', 'text-green', tradingFields)}
        {renderFieldGroup('Other Settings', 'other', 'text-cyan', otherFields)}

        {/* Config Change History */}
        <div className="bg-bg-card rounded-lg border border-border p-5">
          <h3 className="text-sm font-semibold text-text-primary uppercase tracking-wider mb-4 flex items-center gap-2">
            <span className="text-purple">&#9670;</span>
            Change History
          </h3>
          {(!history || history.length === 0) ? (
            <p className="text-xs text-text-muted">No configuration changes recorded</p>
          ) : (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {history.slice(0, 20).map((entry, i) => (
                <div key={i} className="border-l-2 border-l-purple pl-3 py-1">
                  <p className="text-xs text-text-primary">
                    {entry.key as string ?? 'config'}: {String(entry.old_value ?? '')} -&gt; {String(entry.new_value ?? '')}
                  </p>
                  <p className="text-[10px] text-text-muted">
                    {entry.timestamp as string ?? entry.time as string ?? ''}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* System Health */}
      <div className="flex items-center gap-4 px-4 py-3 bg-bg-card rounded-lg border border-border text-[10px]">
        <div className="flex items-center gap-2">
          <span className="text-text-muted">CONFIG SOURCE</span>
          <span className={`w-2 h-2 rounded-full ${configError ? 'bg-red' : 'bg-green'}`} />
          <span className="text-text-primary font-mono">
            {configError ? 'OFFLINE' : 'LIVE'}
          </span>
        </div>
        <span className="text-text-muted">
          FIELDS: <span className="text-text-primary font-mono">{configEntries.length}</span>
        </span>
        <span className="text-text-muted">
          DIRTY: <span className={`font-mono ${dirtyKeys.size > 0 ? 'text-yellow' : 'text-text-primary'}`}>
            {dirtyKeys.size}
          </span>
        </span>
      </div>
    </div>
  )
}
