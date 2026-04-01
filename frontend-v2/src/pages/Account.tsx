import { useState, useEffect, useCallback } from 'react'
import { Trash2, Plus, Check } from 'lucide-react'
import styles from './Account.module.css'

/* ── Types ── */
interface UserProfile {
  user_id: string; email: string; name: string; picture: string
  provider: string; tier: string; is_admin: boolean
  monthly_pnl: number; monthly_pnl_reset_date: string
  created_at: string; last_login: string
}
interface Credential { id: number; broker: string; label: string; is_paper: boolean; is_active: boolean; created_at: string }
interface Subscription { tier: string; status: string; monthly_profit_cap: number | null; limits: TierInfo }
interface TierInfo { monthly_profit_cap: number | null; max_positions: number; brokers: string[]; paper_only: boolean; features: string[] }

const TIER_PRICES: Record<string, string> = { free: '$0', starter: '$29', pro: '$59' }
const TIER_NAMES: Record<string, string> = { free: 'Free', starter: 'Starter', pro: 'Professional' }

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...opts, credentials: 'include', headers: { 'Content-Type': 'application/json', ...(opts?.headers as Record<string, string>) } })
  return res.json()
}

export default function Account() {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [credentials, setCredentials] = useState<Credential[]>([])
  const [subscription, setSubscription] = useState<Subscription | null>(null)
  const [tiers, setTiers] = useState<Record<string, TierInfo>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [showForm, setShowForm] = useState(false)

  const [credBroker, setCredBroker] = useState('alpaca')
  const [credLabel, setCredLabel] = useState('paper')
  const [credKey, setCredKey] = useState('')
  const [credSecret, setCredSecret] = useState('')
  const [credPaper, setCredPaper] = useState(true)
  const [saving, setSaving] = useState(false)

  const load = useCallback(async () => {
    try {
      const [p, c, s, t] = await Promise.all([
        apiFetch<UserProfile>('/api/users/me'),
        apiFetch<Credential[]>('/api/users/me/credentials'),
        apiFetch<Subscription>('/api/users/me/subscription'),
        apiFetch<Record<string, TierInfo>>('/api/admin/tiers'),
      ])
      if ((p as any).error) { setError((p as any).error); return }
      setProfile(p)
      setCredentials(Array.isArray(c) ? c : [])
      setSubscription(s)
      setTiers(t)
    } catch (e: any) { setError(e.message || 'Failed to load') }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const addCred = async () => {
    if (!credKey || !credSecret) return
    setSaving(true)
    try {
      await apiFetch('/api/users/me/credentials', {
        method: 'POST',
        body: JSON.stringify({ broker: credBroker, credentials: { api_key: credKey, secret_key: credSecret }, label: credLabel, is_paper: credPaper }),
      })
      setShowForm(false); setCredKey(''); setCredSecret('')
      load()
    } catch (e: any) { setError(e.message) }
    finally { setSaving(false) }
  }

  const delCred = async (id: number) => {
    await apiFetch(`/api/users/me/credentials/${id}`, { method: 'DELETE' })
    load()
  }

  if (loading) return <div className={styles.page}><div className={styles.loading}>Loading account...</div></div>
  if (error && !profile) return <div className={styles.page}><div className={styles.error}>{error}</div></div>

  const tier = profile?.tier || 'free'
  const cap = subscription?.limits?.monthly_profit_cap
  const pnl = Math.abs(profile?.monthly_pnl || 0)
  const pctUsed = cap ? Math.min((pnl / cap) * 100, 100) : 0

  return (
    <div className={styles.page}>
      <h1 className={styles.heading}>Account</h1>

      {/* ── Profile + Tier ── */}
      <div className={styles.profileRow}>
        <div className={styles.card}>
          <div className={styles.identity}>
            <div className={styles.avatar}>
              {profile?.picture
                ? <img src={profile.picture} alt="" />
                : <span className={styles.avatarLetter}>{(profile?.name || profile?.email || '?')[0].toUpperCase()}</span>}
            </div>
            <div className={styles.nameBlock}>
              <div className={styles.name}>{profile?.name || 'User'}</div>
              <div className={styles.email}>{profile?.email}</div>
              <div className={styles.meta}>
                <span className={styles.metaChip}><span>Provider</span> {profile?.provider || '—'}</span>
                <span className={styles.metaChip}><span>Joined</span> {profile?.created_at ? new Date(profile.created_at).toLocaleDateString() : '—'}</span>
                <span className={styles.metaChip}><span>Last login</span> {profile?.last_login ? new Date(profile.last_login).toLocaleDateString() : '—'}</span>
              </div>
            </div>
          </div>
        </div>
        <div className={`${styles.card} ${styles.tierCard}`}>
          <div className={styles.tierName}>Current Plan</div>
          <div className={styles.tierBadge}>{TIER_NAMES[tier] || tier}<span> / mo</span></div>
          <div className={styles.tierLabel}>{cap ? `$${cap.toLocaleString()} profit cap` : 'Unlimited'}</div>
        </div>
      </div>

      {/* ── Meters ── */}
      <div className={styles.meterRow}>
        <div className={styles.card}>
          <div className={styles.meterLabel}>Monthly P&L</div>
          <div className={styles.meterValue}>${pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
          {cap && (
            <>
              <div className={styles.progressTrack}>
                <div className={`${styles.progressFill} ${pctUsed > 90 ? styles.progressDanger : pctUsed > 70 ? styles.progressWarn : ''}`} style={{ width: `${pctUsed}%` }} />
              </div>
              <div className={styles.meterSub}>{pctUsed.toFixed(0)}% of ${cap.toLocaleString()} cap</div>
            </>
          )}
        </div>
        <div className={styles.card}>
          <div className={styles.meterLabel}>Max Positions</div>
          <div className={styles.meterValue}>{subscription?.limits?.max_positions || '—'}</div>
          <div className={styles.meterSub}>{subscription?.limits?.brokers?.join(', ') || '—'}</div>
        </div>
        <div className={styles.card}>
          <div className={styles.meterLabel}>Features</div>
          <div className={styles.meterValue}>{subscription?.limits?.features?.length || 0}</div>
          <div className={styles.meterSub}>{subscription?.limits?.features?.join(', ') || '—'}</div>
        </div>
      </div>

      {/* ── Broker Credentials ── */}
      <div className={styles.sectionHead}>
        Broker Credentials
        <button className={styles.sectionAction} onClick={() => setShowForm(!showForm)}>
          {showForm ? 'Cancel' : <><Plus size={12} style={{ verticalAlign: -2 }} /> Add credential</>}
        </button>
      </div>

      {showForm && (
        <div className={styles.credForm}>
          <div className={styles.formField}>
            <label className={styles.formLabel}>Broker</label>
            <select className={styles.formSelect} value={credBroker} onChange={e => setCredBroker(e.target.value)}>
              <option value="alpaca">Alpaca</option>
              <option value="oanda">OANDA</option>
              <option value="ibkr">IBKR</option>
            </select>
          </div>
          <div className={styles.formField}>
            <label className={styles.formLabel}>Label</label>
            <input className={styles.formInput} value={credLabel} onChange={e => setCredLabel(e.target.value)} placeholder="e.g. paper, live" />
          </div>
          <div className={styles.formField}>
            <label className={styles.formLabel}>API Key</label>
            <input className={styles.formInput} value={credKey} onChange={e => setCredKey(e.target.value)} placeholder="Enter API key" type="password" />
          </div>
          <div className={styles.formField}>
            <label className={styles.formLabel}>Secret Key</label>
            <input className={styles.formInput} value={credSecret} onChange={e => setCredSecret(e.target.value)} placeholder="Enter secret key" type="password" />
          </div>
          <div className={`${styles.formField} ${styles.formFull}`}>
            <label className={styles.formCheck}>
              <input type="checkbox" checked={credPaper} onChange={e => setCredPaper(e.target.checked)} />
              Paper trading account
            </label>
          </div>
          <div className={styles.formActions}>
            <button className={styles.btnGhost} onClick={() => setShowForm(false)}>Cancel</button>
            <button className={styles.btnAccent} onClick={addCred} disabled={saving}>{saving ? 'Saving...' : 'Save'}</button>
          </div>
        </div>
      )}

      {credentials.length === 0 ? (
        <div className={styles.empty}>No broker credentials stored yet</div>
      ) : (
        <table className={styles.credTable}>
          <thead><tr><th>Broker</th><th>Label</th><th>Mode</th><th>Status</th><th>Added</th><th></th></tr></thead>
          <tbody>
            {credentials.map(c => (
              <tr key={c.id}>
                <td className={styles.credBroker}>{c.broker}</td>
                <td>{c.label}</td>
                <td><span className={`${styles.credBadge} ${c.is_paper ? styles.credPaper : styles.credLive}`}>{c.is_paper ? 'paper' : 'live'}</span></td>
                <td>{c.is_active ? 'Active' : 'Inactive'}</td>
                <td style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: '#555' }}>{c.created_at ? new Date(c.created_at).toLocaleDateString() : '—'}</td>
                <td><button className={styles.credDel} onClick={() => delCred(c.id)} title="Remove"><Trash2 size={14} /></button></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* ── Plans ── */}
      <div className={styles.sectionHead}>Plans</div>
      <div className={styles.tiersRow}>
        {(['free', 'starter', 'pro'] as const).map(t => {
          const info = tiers[t]
          const active = tier === t
          return (
            <div key={t} className={`${styles.tierSlot} ${active ? styles.tierSlotActive : ''}`}>
              {active && <span className={styles.tierCheck}><Check size={16} /></span>}
              <div className={styles.tierSlotName}>{TIER_NAMES[t]}</div>
              <div className={styles.tierSlotPrice}>{TIER_PRICES[t]}<span>/mo</span></div>
              <div className={styles.tierSlotCap}>${info?.monthly_profit_cap?.toLocaleString() || '200'} profit cap</div>
              <ul className={styles.tierSlotFeatures}>
                <li>{info?.max_positions || '—'} max positions</li>
                <li>{info?.brokers?.join(', ') || '—'}</li>
                {info?.paper_only && <li>Paper trading only</li>}
                <li>{info?.features?.join(', ') || '—'}</li>
              </ul>
              <button className={`${styles.tierSlotBtn} ${active ? styles.tierSlotBtnActive : styles.tierSlotBtnUpgrade}`}>
                {active ? 'Current Plan' : 'Upgrade'}
              </button>
            </div>
          )
        })}
      </div>

      {error && <div className={styles.error}>{error}</div>}
    </div>
  )
}
