const API_BASE = ''

function headers(includeApiKey = false): HeadersInit {
  const h: HeadersInit = { 'Content-Type': 'application/json' }
  const key = (import.meta as any).env?.VITE_API_KEY
  if (includeApiKey && key) (h as Record<string, string>)['X-API-Key'] = key
  return h
}

export async function fetchHealth() {
  const r = await fetch(`${API_BASE}/api/health`, { credentials: 'include' })
  return r.json()
}

export async function fetchState() {
  const r = await fetch(`${API_BASE}/api/state`, { credentials: 'include' })
  return r.json()
}

export async function fetchMetrics() {
  const r = await fetch(`${API_BASE}/api/metrics`, { credentials: 'include' })
  return r.json()
}

export async function fetchTrades() {
  const r = await fetch(`${API_BASE}/api/trades`, { credentials: 'include' })
  return r.json()
}

export async function fetchAccount() {
  const r = await fetch(`${API_BASE}/api/account`, { credentials: 'include' })
  return r.json()
}

export async function postScan() {
  const r = await fetch(`${API_BASE}/api/scan`, { method: 'POST', headers: headers(true), credentials: 'include' })
  return r.json()
}

export async function postStart() {
  const r = await fetch(`${API_BASE}/api/start`, { method: 'POST', headers: headers(true), credentials: 'include' })
  return r.json()
}

export async function postStop() {
  const r = await fetch(`${API_BASE}/api/stop`, { method: 'POST', headers: headers(true), credentials: 'include' })
  return r.json()
}

export async function postPause() {
  const r = await fetch(`${API_BASE}/api/pause`, { method: 'POST', headers: headers(true), credentials: 'include' })
  return r.json()
}

export async function postResume() {
  const r = await fetch(`${API_BASE}/api/resume`, { method: 'POST', headers: headers(true), credentials: 'include' })
  return r.json()
}

export async function postReset() {
  const r = await fetch(`${API_BASE}/api/reset`, { method: 'POST', headers: headers(true), credentials: 'include' })
  return r.json()
}

export async function postConfig(body: Record<string, unknown>) {
  const r = await fetch(`${API_BASE}/api/config`, {
    method: 'POST',
    headers: headers(true),
    body: JSON.stringify(body),
    credentials: 'include',
  })
  return r.json()
}

export function wsUrl(): string {
  const base = typeof window !== 'undefined' ? window.location.origin : ''
  const proto = base.startsWith('https') ? 'wss' : 'ws'
  const host = base.replace(/^https?:\/\//, '')
  return `${proto}://${host}/ws`
}
