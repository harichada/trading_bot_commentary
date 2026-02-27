import React, {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useRef,
  useMemo,
} from 'react';
import {AppState, AppStateStatus} from 'react-native';
import {useAuth} from './AuthContext';
import {useWebSocket} from './WebSocketContext';
import type {
  TradingState,
  TraderStatus,
  GapCandidate,
  GapPosition,
  TradeRecord,
  Metrics,
  DailyStats,
  GapFadeConfig,
  LLMStatus,
  Message,
  WSMessage,
  BacktestResult,
} from '../types/trading';

// ─── State ──────────────────────────────────────────────────────

interface AppTradingState extends TradingState {
  connected: boolean;
  backtest: {
    running: boolean;
    progress: number;
    message: string;
    result: BacktestResult | null;
    logs: unknown[];
  };
  trackerPrices: Record<string, {price: number; bid: number; ask: number}>;
  dbProgress: {running: boolean; progress: number; message: string};
}

const initialState: AppTradingState = {
  status: 'stopped',
  equity: 0,
  peak_equity: 0,
  positions: {},
  daily_stats: {
    date: '',
    trades: 0,
    wins: 0,
    losses: 0,
    pnl: 0,
    peak_equity: 0,
    consecutive_losses: 0,
    halted: false,
    halt_reason: '',
  },
  candidates: [],
  last_scan_time: '',
  metrics: {
    total_trades: 0,
    win_rate: 0,
    profit_factor: 0,
    total_pnl: 0,
    avg_pnl: 0,
    avg_winner: 0,
    avg_loser: 0,
    best_trade: 0,
    worst_trade: 0,
    sharpe: 0,
    max_drawdown: 0,
    avg_holding_minutes: 0,
  },
  messages: [],
  config: {} as GapFadeConfig,
  today_trades: [],
  llm: {
    enabled: false,
    available: false,
    model: '',
    circuit_open: false,
    failure_count: 0,
  },
  connected: false,
  backtest: {running: false, progress: 0, message: '', result: null, logs: []},
  trackerPrices: {},
  dbProgress: {running: false, progress: 0, message: ''},
};

// ─── Actions ────────────────────────────────────────────────────

type Action =
  | {type: 'SET_FULL_STATE'; payload: TradingState}
  | {type: 'SET_STATUS'; payload: TraderStatus}
  | {type: 'SET_POSITIONS'; payload: Record<string, GapPosition>}
  | {type: 'SET_CANDIDATES'; payload: GapCandidate[]}
  | {type: 'ADD_TRADES'; payload: TradeRecord[]}
  | {type: 'SET_METRICS'; payload: Metrics}
  | {type: 'SET_DAILY_STATS'; payload: DailyStats}
  | {type: 'SET_CONFIG'; payload: GapFadeConfig}
  | {type: 'SET_LLM'; payload: LLMStatus}
  | {type: 'ADD_MESSAGE'; payload: Message}
  | {type: 'SET_CONNECTED'; payload: boolean}
  | {type: 'UPDATE_TICKER_PRICE'; payload: {symbol: string; price: number; bid: number; ask: number}}
  | {type: 'SET_BACKTEST_PROGRESS'; payload: {pct: number; message: string}}
  | {type: 'SET_BACKTEST_RESULT'; payload: BacktestResult}
  | {type: 'ADD_BT_LOG'; payload: unknown}
  | {type: 'CLEAR_BACKTEST'}
  | {type: 'SET_DB_PROGRESS'; payload: {progress: number; message: string}}
  | {type: 'SET_EQUITY'; payload: {equity: number; peak_equity: number}};

function reducer(state: AppTradingState, action: Action): AppTradingState {
  switch (action.type) {
    case 'SET_FULL_STATE':
      return {
        ...state,
        ...action.payload,
        // Merge metrics with defaults so .toFixed() never hits undefined
        metrics: {...state.metrics, ...(action.payload.metrics ?? {})},
      };
    case 'SET_STATUS':
      return {...state, status: action.payload};
    case 'SET_POSITIONS':
      return {...state, positions: action.payload};
    case 'SET_CANDIDATES':
      return {...state, candidates: action.payload};
    case 'ADD_TRADES':
      return {
        ...state,
        today_trades: [...state.today_trades, ...action.payload],
      };
    case 'SET_METRICS':
      return {...state, metrics: action.payload};
    case 'SET_DAILY_STATS':
      return {...state, daily_stats: action.payload};
    case 'SET_CONFIG':
      return {...state, config: action.payload};
    case 'SET_LLM':
      return {...state, llm: action.payload};
    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [...state.messages.slice(-99), action.payload],
      };
    case 'SET_CONNECTED':
      return {...state, connected: action.payload};
    case 'UPDATE_TICKER_PRICE':
      return {
        ...state,
        trackerPrices: {
          ...state.trackerPrices,
          [action.payload.symbol]: {
            price: action.payload.price,
            bid: action.payload.bid,
            ask: action.payload.ask,
          },
        },
      };
    case 'SET_BACKTEST_PROGRESS':
      return {
        ...state,
        backtest: {
          ...state.backtest,
          running: true,
          progress: action.payload.pct,
          message: action.payload.message,
        },
      };
    case 'SET_BACKTEST_RESULT':
      return {
        ...state,
        backtest: {
          ...state.backtest,
          running: false,
          progress: 100,
          result: action.payload,
        },
      };
    case 'ADD_BT_LOG':
      return {
        ...state,
        backtest: {
          ...state.backtest,
          logs: [...state.backtest.logs, action.payload],
        },
      };
    case 'CLEAR_BACKTEST':
      return {
        ...state,
        backtest: {running: false, progress: 0, message: '', result: null, logs: []},
      };
    case 'SET_DB_PROGRESS':
      return {
        ...state,
        dbProgress: {
          running: action.payload.progress < 100,
          progress: action.payload.progress,
          message: action.payload.message,
        },
      };
    case 'SET_EQUITY':
      return {
        ...state,
        equity: action.payload.equity,
        peak_equity: action.payload.peak_equity,
      };
    default:
      return state;
  }
}

// ─── Context ────────────────────────────────────────────────────

interface TradingStateCtx {
  state: AppTradingState;
  dispatch: React.Dispatch<Action>;
  refreshState: () => Promise<void>;
}

const TradingStateContext = createContext<TradingStateCtx | null>(null);

const POLL_INTERVAL_CONNECTED = 30_000;
const POLL_INTERVAL_DISCONNECTED = 5_000;

export function TradingStateProvider({children}: {children: React.ReactNode}) {
  const {apiClient} = useAuth();
  const {connected, subscribe} = useWebSocket();
  const [state, dispatch] = useReducer(reducer, initialState);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Sync WS connected flag into state
  useEffect(() => {
    dispatch({type: 'SET_CONNECTED', payload: connected});
  }, [connected]);

  // Fetch full state
  const refreshState = async () => {
    try {
      const data = await apiClient.getState();
      dispatch({type: 'SET_FULL_STATE', payload: data});
    } catch {
      // Silently fail — will retry on next poll
    }
  };

  // Initial load + polling
  useEffect(() => {
    refreshState();

    const interval = connected
      ? POLL_INTERVAL_CONNECTED
      : POLL_INTERVAL_DISCONNECTED;

    pollRef.current = setInterval(refreshState, interval);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connected, apiClient]);

  // Pause polling when app backgrounds
  useEffect(() => {
    const handler = (appState: AppStateStatus) => {
      if (appState === 'active') {
        refreshState();
      }
    };
    const sub = AppState.addEventListener('change', handler);
    return () => sub.remove();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiClient]);

  // Subscribe to WebSocket messages
  useEffect(() => {
    const unsub = subscribe((msg: WSMessage) => {
      switch (msg.type) {
        case 'live_status':
          dispatch({type: 'SET_STATUS', payload: msg.status as string as any});
          break;
        case 'trade':
          dispatch({type: 'ADD_TRADES', payload: (msg as any).trades});
          break;
        case 'positions_update':
          if ((msg as any).positions) {
            dispatch({type: 'SET_POSITIONS', payload: (msg as any).positions});
          }
          if ((msg as any).equity != null) {
            dispatch({
              type: 'SET_EQUITY',
              payload: {
                equity: (msg as any).equity,
                peak_equity: (msg as any).peak_equity ?? state.peak_equity,
              },
            });
          }
          break;
        case 'scan_results':
          dispatch({type: 'SET_CANDIDATES', payload: (msg as any).candidates});
          break;
        case 'tracker_tick':
          dispatch({
            type: 'UPDATE_TICKER_PRICE',
            payload: {
              symbol: (msg as any).symbol,
              price: (msg as any).price,
              bid: (msg as any).bid ?? 0,
              ask: (msg as any).ask ?? 0,
            },
          });
          break;
        case 'backtest_progress':
          dispatch({
            type: 'SET_BACKTEST_PROGRESS',
            payload: {pct: (msg as any).pct, message: (msg as any).message},
          });
          break;
        case 'backtest_complete':
          dispatch({type: 'SET_BACKTEST_RESULT', payload: (msg as any).result});
          break;
        case 'bt_log':
          dispatch({type: 'ADD_BT_LOG', payload: (msg as any).entry});
          break;
        case 'db_build_progress':
          dispatch({
            type: 'SET_DB_PROGRESS',
            payload: {
              progress: (msg as any).progress ?? 0,
              message: (msg as any).message ?? '',
            },
          });
          break;
      }
    });
    return unsub;
  }, [subscribe, state.peak_equity]);

  const value = useMemo(
    () => ({state, dispatch, refreshState}),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [state],
  );

  return (
    <TradingStateContext.Provider value={value}>
      {children}
    </TradingStateContext.Provider>
  );
}

export function useTradingState(): TradingStateCtx {
  const ctx = useContext(TradingStateContext);
  if (!ctx) throw new Error('useTradingState must be inside TradingStateProvider');
  return ctx;
}
