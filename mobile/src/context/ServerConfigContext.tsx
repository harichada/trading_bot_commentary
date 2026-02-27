import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
} from 'react';
import * as SecureStore from 'expo-secure-store';

const STORAGE_KEY_URL = 'gapfade_serverUrl';
const STORAGE_KEY_API_KEY = 'gapfade_apiKey';

interface ServerConfig {
  serverUrl: string;
  apiKey: string;
  loading: boolean;
  setServerUrl: (url: string) => Promise<void>;
  setApiKey: (key: string) => Promise<void>;
  clearConfig: () => Promise<void>;
}

const ServerConfigContext = createContext<ServerConfig | null>(null);

export function ServerConfigProvider({children}: {children: React.ReactNode}) {
  const [serverUrl, setUrlState] = useState('');
  const [apiKey, setKeyState] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      try {
        const [url, key] = await Promise.all([
          SecureStore.getItemAsync(STORAGE_KEY_URL),
          SecureStore.getItemAsync(STORAGE_KEY_API_KEY),
        ]);
        if (url) setUrlState(url);
        if (key) setKeyState(key);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const setServerUrl = useCallback(async (url: string) => {
    const trimmed = url.replace(/\/+$/, '');
    await SecureStore.setItemAsync(STORAGE_KEY_URL, trimmed);
    setUrlState(trimmed);
  }, []);

  const setApiKey = useCallback(async (key: string) => {
    await SecureStore.setItemAsync(STORAGE_KEY_API_KEY, key);
    setKeyState(key);
  }, []);

  const clearConfig = useCallback(async () => {
    await Promise.all([
      SecureStore.deleteItemAsync(STORAGE_KEY_URL),
      SecureStore.deleteItemAsync(STORAGE_KEY_API_KEY),
    ]);
    setUrlState('');
    setKeyState('');
  }, []);

  const value = useMemo(
    () => ({serverUrl, apiKey, loading, setServerUrl, setApiKey, clearConfig}),
    [serverUrl, apiKey, loading, setServerUrl, setApiKey, clearConfig],
  );

  return (
    <ServerConfigContext.Provider value={value}>
      {children}
    </ServerConfigContext.Provider>
  );
}

export function useServerConfig(): ServerConfig {
  const ctx = useContext(ServerConfigContext);
  if (!ctx) throw new Error('useServerConfig must be inside ServerConfigProvider');
  return ctx;
}
