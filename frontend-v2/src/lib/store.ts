/** Global reactive store — WebSocket-driven state shared across all pages */

type Listener = () => void

interface Store {
  // State
  traderStatus: string
  equity: number
  dailyPnl: number
  positions: Record<string, unknown>[]
  candidates: Record<string, unknown>[]
  messages: Record<string, unknown>[]
  feed: { time: string; level: string; text: string }[]
  connected: boolean
  wsLatencyMs: number
  lastTradeTime: string
  // Backtest
  btProgress: { msg: string; pct: number } | null
  btComplete: Record<string, unknown> | null
  wfComplete: Record<string, unknown> | null
  iwfComplete: Record<string, unknown> | null
  dbProgress: { msg: string; pct: number } | null
  // Config
  configChanged: Record<string, unknown> | null
  strategySwitched: string | null
  // Tracker ticks
  trackerTicks: Record<string, { price: number; time: number }>
}

const state: Store = {
  traderStatus: 'stopped', equity: 0, dailyPnl: 0,
  positions: [], candidates: [], messages: [], feed: [],
  connected: false, wsLatencyMs: 0, lastTradeTime: '',
  btProgress: null, btComplete: null, wfComplete: null, iwfComplete: null, dbProgress: null,
  configChanged: null, strategySwitched: null, trackerTicks: {},
}

const listeners = new Set<Listener>()
let ws: WebSocket | null = null
let pingTime = 0

function notify() { listeners.forEach(fn => fn()) }

export function getStore(): Readonly<Store> { return state }

export function subscribe(fn: Listener): () => void {
  listeners.add(fn)
  return () => listeners.delete(fn)
}

export function addFeedItem(level: string, text: string) {
  const now = new Date()
  const time = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  state.feed = [{ time, level, text }, ...state.feed.slice(0, 499)]
  notify()
}

function handleMessage(msg: Record<string, unknown>) {
  const type = msg.type as string

  switch (type) {
    case 'pong':
      state.wsLatencyMs = Date.now() - pingTime
      break

    case 'live_status':
      state.traderStatus = String(msg.status ?? state.traderStatus)
      if (msg.equity != null) state.equity = Number(msg.equity)
      addFeedItem('status', `Status: ${state.traderStatus}`)
      break

    case 'positions_update':
      state.positions = Array.isArray(msg.positions) ? msg.positions : Object.values(msg.positions ?? {})
      break

    case 'scan_results':
      state.candidates = (msg.candidates as Record<string, unknown>[]) ?? []
      addFeedItem('scan', `Scan: ${state.candidates.length} candidates`)
      break

    case 'trade':
      state.lastTradeTime = new Date().toLocaleTimeString('en-US', { hour12: false })
      addFeedItem('trade', `Trade executed`)
      break

    case 'backtest_progress':
      state.btProgress = { msg: String(msg.message ?? ''), pct: Number(msg.progress ?? 0) }
      break

    case 'backtest_complete':
      state.btComplete = msg
      state.btProgress = null
      break

    case 'walkforward_complete':
      state.wfComplete = msg
      break

    case 'intraday_walkforward_complete':
      state.iwfComplete = msg
      break

    case 'db_build_progress':
      state.dbProgress = { msg: String(msg.message ?? ''), pct: Number(msg.progress ?? 0) }
      if (Number(msg.progress ?? 0) >= 100) {
        setTimeout(() => { state.dbProgress = null; notify() }, 2000)
      }
      break

    case 'tracker_tick': {
      const sym = String(msg.symbol ?? '')
      if (sym) {
        state.trackerTicks = { ...state.trackerTicks, [sym]: { price: Number(msg.price ?? 0), time: Date.now() } }
      }
      break
    }

    case 'conversation':
      addFeedItem('llm', `${msg.speaker ?? 'BOT'}: ${String(msg.text ?? '').slice(0, 200)}`)
      break

    case 'config_changed':
      state.configChanged = msg
      addFeedItem('config', 'Config changed')
      break

    case 'strategy_switched':
      state.strategySwitched = String(msg.active ?? '')
      addFeedItem('strategy', `Strategy: ${state.strategySwitched}`)
      break

    case 'bt_log':
      addFeedItem('bt', String(msg.message ?? msg.text ?? ''))
      break

    default:
      // Feed any unhandled message types
      if (msg.text || msg.message) {
        addFeedItem(type || 'info', String(msg.text ?? msg.message ?? ''))
      }
  }

  notify()
}

export function connectWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return

  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const apiKey = localStorage.getItem('rudra_api_key') ?? ''
  const wsUrl = apiKey
    ? `${proto}://${location.host}/ws?apiKey=${encodeURIComponent(apiKey)}`
    : `${proto}://${location.host}/ws`
  ws = new WebSocket(wsUrl)

  ws.onopen = () => {
    state.connected = true
    notify()
  }

  ws.onclose = () => {
    state.connected = false
    notify()
    setTimeout(connectWebSocket, 2000)
  }

  ws.onerror = () => { /* swallow */ }

  ws.onmessage = (e) => {
    try {
      handleMessage(JSON.parse(e.data))
    } catch { /* ignore */ }
  }

  // Keepalive ping every 15s
  setInterval(() => {
    if (ws?.readyState === WebSocket.OPEN) {
      pingTime = Date.now()
      ws.send(JSON.stringify({ type: 'ping' }))
    }
  }, 15000)
}
