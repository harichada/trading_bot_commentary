/**
 * Live bot data layer.
 *
 * In production the static export is served BY the FastAPI bot, so calls are
 * same-origin ("") and the API key is injected into <head> as
 * window.TRADING_API_KEY (exactly like the legacy dashboard.html authFetch).
 * For local dev against a bot on another origin, set NEXT_PUBLIC_BOT_URL
 * (e.g. http://localhost:9000) — note the bot must allow CORS for that origin.
 */

const BOT_URL = process.env.NEXT_PUBLIC_BOT_URL ?? ""

function getApiKey(): string {
  if (typeof window === "undefined") return ""
  return (window as unknown as { TRADING_API_KEY?: string }).TRADING_API_KEY ?? ""
}

export async function authFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getApiKey()
  const headers = new Headers(options.headers)
  if (token) headers.set("Authorization", `Bearer ${token}`)
  const res = await fetch(`${BOT_URL}${path}`, { ...options, headers })
  if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`)
  return (await res.json()) as T
}

/** Build the authenticated WebSocket URL: ws(s)://host/ws?token=KEY */
export function botWsUrl(): string {
  const token = getApiKey()
  let base: string
  if (BOT_URL) {
    base = BOT_URL.replace(/^http/, "ws")
  } else if (typeof window !== "undefined") {
    const proto = window.location.protocol === "https:" ? "wss" : "ws"
    base = `${proto}://${window.location.host}`
  } else {
    base = "ws://localhost:9000"
  }
  return `${base}/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`
}

// ── Typed response shapes (verified against the live bot 2026-06-23) ──────────

export type AccountSource = "schwab" | "simulation" | "internal" | "none"

export interface AccountStats {
  status: string
  source: AccountSource
  stats: {
    balance: number
    buying_power: number
    daily_pnl: number
    total_pnl: number
    position_count: number
    cash: number
  }
}

export interface BotStatus {
  status: string
  bot_running: boolean
  mode: string
  positions_count: number
  schwab_connected: boolean
  uptime: string
}

export interface MarketIndex {
  name?: string
  symbol?: string
  value?: number
  price?: number
  change?: number
  change_percent?: number
  changePercent?: number
}

export interface MarketIndices {
  indices: MarketIndex[]
  stale?: boolean
}

export const fetchAccountStats = () => authFetch<AccountStats>("/api/account-stats")
export const fetchBotStatus = () => authFetch<BotStatus>("/api/status")
export const fetchMarketIndices = () => authFetch<MarketIndices>("/api/market-indices")

/** Whether the account source represents real brokerage data. */
export const isLiveSource = (s: AccountSource) => s === "schwab"
