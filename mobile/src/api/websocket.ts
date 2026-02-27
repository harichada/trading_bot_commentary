import type {WSMessage} from '../types/trading';

export type WSListener = (message: WSMessage) => void;
export type ConnectionListener = (connected: boolean) => void;

const HEARTBEAT_INTERVAL = 30_000;
const RECONNECT_BASE = 1_000;
const RECONNECT_MAX = 30_000;

export class GapFadeWebSocket {
  private ws: WebSocket | null = null;
  private heartbeat: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay = RECONNECT_BASE;
  private listeners = new Set<WSListener>();
  private connectionListeners = new Set<ConnectionListener>();
  private _connected = false;
  private shouldConnect = false;

  private authToken: string = '';

  constructor(
    private serverUrl: string,
    private apiKey: string,
  ) {}

  setAuthToken(token: string) {
    this.authToken = token;
  }

  get connected(): boolean {
    return this._connected;
  }

  updateConfig(url: string, key: string) {
    const changed = url !== this.serverUrl || key !== this.apiKey;
    this.serverUrl = url;
    this.apiKey = key;
    if (changed && this.shouldConnect) {
      this.disconnect();
      this.connect();
    }
  }

  connect() {
    if (this.ws) {
      return;
    }
    this.shouldConnect = true;

    const proto = this.serverUrl.startsWith('https') ? 'wss' : 'ws';
    const host = this.serverUrl.replace(/^https?:\/\//, '');
    const params = new URLSearchParams();
    if (this.authToken) params.set('token', this.authToken);
    if (this.apiKey) params.set('apiKey', this.apiKey);
    const qs = params.toString();
    const url = `${proto}://${host}/ws${qs ? `?${qs}` : ''}`;

    try {
      this.ws = new WebSocket(url);
    } catch {
      this.scheduleReconnect();
      return;
    }

    this.ws.onopen = () => {
      this._connected = true;
      this.reconnectDelay = RECONNECT_BASE;
      this.notifyConnection(true);
      this.startHeartbeat();
    };

    this.ws.onmessage = (event: WebSocketMessageEvent) => {
      const data = event.data;
      if (data === 'pong') {
        return;
      }
      try {
        const msg: WSMessage = JSON.parse(data as string);
        this.notifyListeners(msg);
      } catch {
        // Ignore non-JSON messages
      }
    };

    this.ws.onclose = () => {
      this.cleanup();
      this.notifyConnection(false);
      if (this.shouldConnect) {
        this.scheduleReconnect();
      }
    };

    this.ws.onerror = () => {
      // onclose will fire after onerror
    };
  }

  disconnect() {
    this.shouldConnect = false;
    this.cleanup();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.notifyConnection(false);
  }

  onMessage(listener: WSListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onConnection(listener: ConnectionListener): () => void {
    this.connectionListeners.add(listener);
    return () => this.connectionListeners.delete(listener);
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeat = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send('ping');
      }
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat() {
    if (this.heartbeat) {
      clearInterval(this.heartbeat);
      this.heartbeat = null;
    }
  }

  private cleanup() {
    this.stopHeartbeat();
    this._connected = false;
    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onclose = null;
      this.ws.onerror = null;
      if (
        this.ws.readyState === WebSocket.OPEN ||
        this.ws.readyState === WebSocket.CONNECTING
      ) {
        this.ws.close();
      }
      this.ws = null;
    }
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) {
      return;
    }
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.reconnectDelay = Math.min(
        this.reconnectDelay * 2,
        RECONNECT_MAX,
      );
      this.ws = null;
      this.connect();
    }, this.reconnectDelay);
  }

  private notifyListeners(msg: WSMessage) {
    this.listeners.forEach(fn => fn(msg));
  }

  private notifyConnection(connected: boolean) {
    this.connectionListeners.forEach(fn => fn(connected));
  }
}
