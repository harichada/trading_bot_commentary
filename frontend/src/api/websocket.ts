type MessageHandler = (data: Record<string, unknown>) => void

interface WSManagerOptions {
  url: string
  onMessage: MessageHandler
  onStatusChange?: (connected: boolean) => void
  reconnectInterval?: number
}

export class WSManager {
  private ws: WebSocket | null = null
  private readonly url: string
  private readonly onMessage: MessageHandler
  private readonly onStatusChange?: (connected: boolean) => void
  private readonly reconnectInterval: number
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private intentionalClose = false

  constructor(opts: WSManagerOptions) {
    this.url = opts.url
    this.onMessage = opts.onMessage
    this.onStatusChange = opts.onStatusChange
    this.reconnectInterval = opts.reconnectInterval ?? 5000
  }

  connect(): void {
    this.intentionalClose = false
    try {
      this.ws = new WebSocket(this.url)
    } catch {
      this.scheduleReconnect()
      return
    }

    this.ws.onopen = () => {
      this.onStatusChange?.(true)
    }

    this.ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as Record<string, unknown>
        this.onMessage(data)
      } catch {
        // ignore non-JSON messages
      }
    }

    this.ws.onclose = () => {
      this.onStatusChange?.(false)
      if (!this.intentionalClose) {
        this.scheduleReconnect()
      }
    }

    this.ws.onerror = () => {
      this.ws?.close()
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimer) return
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connect()
    }, this.reconnectInterval)
  }

  disconnect(): void {
    this.intentionalClose = true
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    this.ws?.close()
    this.ws = null
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN
  }
}

function wsUrl(path: string): string {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${proto}//${window.location.host}${path}`
}

export function createGapFadeWS(onMessage: MessageHandler, onStatusChange?: (connected: boolean) => void): WSManager {
  return new WSManager({
    url: wsUrl('/gf-ws'),
    onMessage,
    onStatusChange,
  })
}

export function createIntradayWS(onMessage: MessageHandler, onStatusChange?: (connected: boolean) => void): WSManager {
  return new WSManager({
    url: wsUrl('/id-ws'),
    onMessage,
    onStatusChange,
  })
}
