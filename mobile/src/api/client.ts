import type {
  TradingState,
  GapFadeConfig,
  TradeRecord,
  HealthResponse,
  AccountInfo,
  TrackerBar,
  DBStats,
  BacktestResult,
  LLMStatus,
  Metrics,
} from '../types/trading';

export class ApiError extends Error {
  constructor(
    public status: number,
    public body: unknown,
  ) {
    super(`API error ${status}`);
  }
}

export class GapFadeApiClient {
  private authToken: string = '';

  constructor(
    private baseUrl: string,
    private apiKey: string,
  ) {}

  updateConfig(url: string, key: string) {
    this.baseUrl = url.replace(/\/+$/, '');
    this.apiKey = key;
  }

  setAuthToken(token: string) {
    this.authToken = token;
  }

  private async request<T = unknown>(
    path: string,
    options: RequestInit = {},
  ): Promise<T> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }

    const url = `${this.baseUrl}${path}`;
    const response = await fetch(url, {
      ...options,
      headers: {...headers, ...(options.headers as Record<string, string>)},
    });

    if (!response.ok) {
      let body: unknown;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      throw new ApiError(response.status, body);
    }

    return response.json() as Promise<T>;
  }

  // ─── Health ────────────────────────────────────────────────────
  getHealth(): Promise<HealthResponse> {
    return this.request('/api/health');
  }

  // ─── State ─────────────────────────────────────────────────────
  getState(): Promise<TradingState> {
    return this.request('/api/state');
  }

  getMetrics(): Promise<Metrics> {
    return this.request('/api/metrics');
  }

  getTrades(): Promise<TradeRecord[]> {
    return this.request('/api/trades');
  }

  getAccount(): Promise<AccountInfo> {
    return this.request('/api/account');
  }

  // ─── Trading Control ──────────────────────────────────────────
  scan(): Promise<{candidates: number}> {
    return this.request('/api/scan', {method: 'POST'});
  }

  start(): Promise<{status: string}> {
    return this.request('/api/start', {method: 'POST'});
  }

  stop(): Promise<{status: string}> {
    return this.request('/api/stop', {method: 'POST'});
  }

  pause(): Promise<{status: string}> {
    return this.request('/api/pause', {method: 'POST'});
  }

  resume(): Promise<{status: string}> {
    return this.request('/api/resume', {method: 'POST'});
  }

  reset(): Promise<{status: string}> {
    return this.request('/api/reset', {method: 'POST'});
  }

  enter(body?: {
    symbols?: string[];
    size_mult?: number;
  }): Promise<{entered: number}> {
    return this.request('/api/enter', {
      method: 'POST',
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  // ─── Position Management ──────────────────────────────────────
  closePosition(symbol: string): Promise<{status: string}> {
    return this.request(`/api/positions/${encodeURIComponent(symbol)}/close`, {
      method: 'POST',
    });
  }

  adjustStop(
    symbol: string,
    stopPrice: number,
  ): Promise<{status: string}> {
    return this.request(`/api/positions/${encodeURIComponent(symbol)}/stop`, {
      method: 'POST',
      body: JSON.stringify({stop_price: stopPrice}),
    });
  }

  // ─── Configuration ────────────────────────────────────────────
  updateStrategyConfig(
    config: Partial<GapFadeConfig>,
  ): Promise<{status: string}> {
    return this.request('/api/config', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  // ─── Backtesting ──────────────────────────────────────────────
  runBacktest(params: {
    symbols?: string;
    start_date?: string;
    end_date?: string;
    config?: Partial<GapFadeConfig>;
  }): Promise<{status: string}> {
    return this.request('/api/backtest', {
      method: 'POST',
      body: JSON.stringify(params),
    });
  }

  getBacktestStatus(): Promise<{
    running: boolean;
    progress: number;
    message: string;
  }> {
    return this.request('/api/backtest/status');
  }

  getBacktestResults(): Promise<BacktestResult> {
    return this.request('/api/backtest/results');
  }

  cancelBacktest(): Promise<{status: string}> {
    return this.request('/api/backtest/cancel', {method: 'POST'});
  }

  // ─── Database ─────────────────────────────────────────────────
  getDBStats(): Promise<DBStats> {
    return this.request('/api/db/stats');
  }

  buildDB(params?: {
    start_date?: string;
    symbols?: string;
  }): Promise<{status: string}> {
    return this.request('/api/db/build', {
      method: 'POST',
      body: params ? JSON.stringify(params) : undefined,
    });
  }

  updateDB(): Promise<{status: string}> {
    return this.request('/api/db/update', {method: 'POST'});
  }

  polygonUpdate(params?: {
    start_date?: string;
  }): Promise<{status: string}> {
    return this.request('/api/db/polygon-update', {
      method: 'POST',
      body: params ? JSON.stringify(params) : undefined,
    });
  }

  // ─── Tracker ──────────────────────────────────────────────────
  async getTrackerBars(params: {
    symbol: string;
    timeframe?: string;
    limit?: number;
  }): Promise<TrackerBar[]> {
    const qs = new URLSearchParams({
      symbol: params.symbol,
      ...(params.timeframe && {timeframe: params.timeframe}),
      ...(params.limit && {limit: String(params.limit)}),
    });
    const res = await this.request<{bars?: Array<{t: number; o: number; h: number; l: number; c: number; v: number}>}>(`/api/tracker/bars?${qs}`);
    const raw = res.bars ?? [];
    return raw.map(b => ({
      timestamp: new Date(b.t * 1000).toISOString(),
      open: b.o,
      high: b.h,
      low: b.l,
      close: b.c,
      volume: b.v,
    }));
  }

  getTrackerPositions(): Promise<{
    positions: unknown[];
    account: AccountInfo;
  }> {
    return this.request('/api/tracker/positions');
  }

  placeOrder(body: {
    symbol: string;
    qty: number;
    side: 'buy' | 'sell';
  }): Promise<{order_id: string}> {
    return this.request('/api/tracker/order', {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  updateWatchlist(symbols: string[]): Promise<{status: string}> {
    return this.request('/api/tracker/watchlist', {
      method: 'POST',
      body: JSON.stringify({symbols}),
    });
  }

  // ─── LLM Supervisor ──────────────────────────────────────────
  getLLMStatus(): Promise<LLMStatus> {
    return this.request('/api/llm/status');
  }

  testLLM(): Promise<{status: string; response: string}> {
    return this.request('/api/llm/test', {method: 'POST'});
  }

  chatLLM(message: string): Promise<{
    response: string;
    action?: string;
    action_result?: unknown;
  }> {
    return this.request('/api/llm/chat', {
      method: 'POST',
      body: JSON.stringify({message}),
    });
  }

  // ─── Auth ─────────────────────────────────────────────────────
  getAuthMe(): Promise<{
    user: {email: string; name: string; picture: string; provider: string} | null;
    auth_enabled: boolean;
  }> {
    return this.request('/api/auth/me');
  }
}
