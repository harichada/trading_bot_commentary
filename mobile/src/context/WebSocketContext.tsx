import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from 'react';
import {AppState, AppStateStatus} from 'react-native';
import * as SecureStore from 'expo-secure-store';
import {useServerConfig} from './ServerConfigContext';
import {useAuth} from './AuthContext';
import {GapFadeWebSocket, WSListener} from '../api/websocket';

const TOKEN_STORAGE_KEY = 'gapfade_authToken';

interface WSContext {
  connected: boolean;
  subscribe: (listener: WSListener) => () => void;
}

const WebSocketContext = createContext<WSContext | null>(null);

export function WebSocketProvider({children}: {children: React.ReactNode}) {
  const {serverUrl, apiKey} = useServerConfig();
  const {user} = useAuth();
  const wsRef = useRef<GapFadeWebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!serverUrl) return;

    const ws = new GapFadeWebSocket(serverUrl, apiKey);
    wsRef.current = ws;

    // Load auth token for WS connection
    (async () => {
      const token = await SecureStore.getItemAsync(TOKEN_STORAGE_KEY);
      if (token) ws.setAuthToken(token);
      const unsub = ws.onConnection(setConnected);
      ws.connect();

      // Store unsub for cleanup
      (ws as any)._connUnsub = unsub;
    })();

    return () => {
      const unsub = (ws as any)._connUnsub;
      if (unsub) unsub();
      ws.disconnect();
      wsRef.current = null;
    };
  }, [serverUrl, apiKey, user]);

  // Pause/resume on app state changes
  useEffect(() => {
    const handler = (state: AppStateStatus) => {
      const ws = wsRef.current;
      if (!ws) return;
      if (state === 'active') {
        ws.connect();
      } else {
        ws.disconnect();
      }
    };
    const sub = AppState.addEventListener('change', handler);
    return () => sub.remove();
  }, []);

  const subscribe = useCallback((listener: WSListener) => {
    const ws = wsRef.current;
    if (!ws) return () => {};
    return ws.onMessage(listener);
  }, []);

  const value = useMemo(() => ({connected, subscribe}), [connected, subscribe]);

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocket(): WSContext {
  const ctx = useContext(WebSocketContext);
  if (!ctx) throw new Error('useWebSocket must be inside WebSocketProvider');
  return ctx;
}
