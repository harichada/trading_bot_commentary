import { useState, useEffect, useCallback } from 'react'
import styles from './Admin.module.css'

interface AdminUser {
  user_id: string; email: string; name: string; provider: string
  tier: string; is_active: boolean; is_admin: boolean
  monthly_pnl: number; created_at: string; last_login: string
}

interface EngineInfo { status: string; positions: number; email: string; tier: string }
interface EnginesData { engines: Record<string, EngineInfo>; count: number; max: number }

const TIERS = ['free', 'starter', 'pro', 'enterprise']

async function apiFetch<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(path, { ...opts, headers: { 'Content-Type': 'application/json', ...(opts?.headers as Record<string, string>) } })
  return res.json()
}

export default function Admin() {
  const [users, setUsers] = useState<AdminUser[]>([])
  const [engines, setEngines] = useState<EnginesData>({ engines: {}, count: 0, max: 0 })
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [noAccess, setNoAccess] = useState(false)

  const load = useCallback(async () => {
    try {
      const [userData, engData] = await Promise.all([
        apiFetch<AdminUser[] | { error: string }>('/api/admin/users'),
        apiFetch<EnginesData>('/api/admin/engines').catch(() => ({ engines: {}, count: 0, max: 0 })),
      ])
      if (!Array.isArray(userData)) {
        if ((userData as any).error?.includes('Admin') || (userData as any).error?.includes('403')) { setNoAccess(true) }
        else { setError((userData as any).error || 'Failed to load') }
        return
      }
      setUsers(userData)
      if (engData && typeof engData === 'object' && 'count' in engData) setEngines(engData)
    } catch (e: any) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  const changeTier = async (userId: string, tier: string) => {
    const res = await apiFetch<{ updated?: boolean; error?: string }>(`/api/admin/users/${userId}/tier`, {
      method: 'PUT', body: JSON.stringify({ tier }),
    })
    if (res.error) { setError(res.error); return }
    setUsers(prev => prev.map(u => u.user_id === userId ? { ...u, tier } : u))
  }

  if (noAccess) return <div className={styles.page}><div className={styles.noAccess}>Admin access required</div></div>
  if (loading) return <div className={styles.page}><div className={styles.loading}>Loading...</div></div>

  const tierCounts = users.reduce<Record<string, number>>((a, u) => { a[u.tier] = (a[u.tier] || 0) + 1; return a }, {})
  const activeCount = users.filter(u => u.is_active).length

  return (
    <div className={styles.page}>
      <h1 className={styles.heading}>Users</h1>

      {error && <div className={styles.error}>{error}</div>}

      {/* ── Stats ── */}
      <div className={styles.statsRow}>
        <div className={styles.stat}>
          <div className={styles.statLabel}>Total</div>
          <div className={styles.statValue}>{users.length}</div>
        </div>
        <div className={styles.stat}>
          <div className={styles.statLabel}>Active</div>
          <div className={`${styles.statValue} ${styles.statAccent}`}>{activeCount}</div>
        </div>
        {TIERS.map(t => (
          <div key={t} className={styles.stat}>
            <div className={styles.statLabel}>{t}</div>
            <div className={styles.statValue}>{tierCounts[t] || 0}</div>
          </div>
        ))}
      </div>

      {/* ── Engines ── */}
      {engines.max > 0 && (
        <div style={{ padding: '16px 0 0', borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 12 }}>
            <span style={{ fontSize: 15, fontWeight: 600, color: '#fff' }}>Trading Engines</span>
            <span style={{ fontSize: 12, color: '#555', fontFamily: 'var(--font-mono)' }}>
              {engines.count} / {engines.max} active
            </span>
          </div>
          {Object.keys(engines.engines).length > 0 ? (
            <table className={styles.table} style={{ marginBottom: 16 }}>
              <thead><tr><th>User</th><th>Tier</th><th>Status</th><th>Positions</th></tr></thead>
              <tbody>
                {Object.entries(engines.engines).map(([uid, eng]) => (
                  <tr key={uid}>
                    <td className={styles.cellEmail}>{eng.email}</td>
                    <td><span className={styles.cellProvider}>{eng.tier}</span></td>
                    <td>
                      <span className={`${styles.statusDot} ${eng.status === 'running' || eng.status === 'scanning' ? styles.dotActive : styles.dotInactive}`} />
                      {eng.status}
                    </td>
                    <td className={styles.cellMono}>{eng.positions}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div style={{ padding: '12px 0 16px', color: '#444', fontSize: 13 }}>No user engines running</div>
          )}
        </div>
      )}

      {/* ── Table ── */}
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>User</th>
              <th>Provider</th>
              <th>Tier</th>
              <th>Status</th>
              <th>Monthly P&L</th>
              <th>Last Login</th>
              <th>Joined</th>
            </tr>
          </thead>
          <tbody>
            {users.map(u => {
              const pnl = u.monthly_pnl || 0
              return (
                <tr key={u.user_id}>
                  <td>
                    <span className={styles.cellEmail}>{u.email}</span>
                    {u.is_admin && <span className={styles.adminTag}>admin</span>}
                    {u.name && <div className={styles.cellName}>{u.name}</div>}
                  </td>
                  <td><span className={styles.cellProvider}>{u.provider || '—'}</span></td>
                  <td>
                    <select className={styles.tierSelect} value={u.tier} onChange={e => changeTier(u.user_id, e.target.value)}>
                      {TIERS.map(t => <option key={t} value={t}>{t}</option>)}
                    </select>
                  </td>
                  <td>
                    <span className={`${styles.statusDot} ${u.is_active ? styles.dotActive : styles.dotInactive}`} />
                    {u.is_active ? 'Active' : 'Inactive'}
                  </td>
                  <td className={`${styles.cellMono} ${pnl >= 0 ? styles.pnlUp : styles.pnlDown}`}>
                    ${pnl.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                  </td>
                  <td className={styles.cellMono}>{u.last_login ? new Date(u.last_login).toLocaleDateString() : '—'}</td>
                  <td className={styles.cellMono}>{u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
