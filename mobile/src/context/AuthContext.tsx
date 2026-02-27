import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
} from 'react';
import {Linking} from 'react-native';
import * as ExpoLinking from 'expo-linking';
import * as WebBrowser from 'expo-web-browser';
import * as SecureStore from 'expo-secure-store';
import {useServerConfig} from './ServerConfigContext';
import {GapFadeApiClient} from '../api/client';

const TOKEN_STORAGE_KEY = 'gapfade_authToken';

interface User {
  email: string;
  name: string;
  picture: string;
  provider: string;
}

interface AuthState {
  user: User | null;
  authEnabled: boolean;
  loading: boolean;
  refresh: () => Promise<void>;
  loginWithProvider: (provider: 'google' | 'github' | 'discord') => Promise<void>;
  logout: () => Promise<void>;
  apiClient: GapFadeApiClient;
}

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({children}: {children: React.ReactNode}) {
  const {serverUrl, apiKey, loading: configLoading} = useServerConfig();
  const [user, setUser] = useState<User | null>(null);
  const [authEnabled, setAuthEnabled] = useState(false);
  const [loading, setLoading] = useState(true);

  const apiClient = useMemo(
    () => new GapFadeApiClient(serverUrl, apiKey),
    [serverUrl, apiKey],
  );

  // Restore saved token on mount
  useEffect(() => {
    (async () => {
      const saved = await SecureStore.getItemAsync(TOKEN_STORAGE_KEY);
      if (saved) {
        apiClient.setAuthToken(saved);
      }
    })();
  }, [apiClient]);

  const refresh = useCallback(async () => {
    if (!serverUrl) {
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      const res = await apiClient.getAuthMe();
      if (res.user) {
        setUser(res.user);
      } else {
        setUser(null);
      }
      setAuthEnabled(res.auth_enabled);
    } catch {
      // 401 means auth is enabled but we're not logged in
      setUser(null);
      setAuthEnabled(true);
    } finally {
      setLoading(false);
    }
  }, [serverUrl, apiClient]);

  useEffect(() => {
    if (!configLoading) {
      refresh();
    }
  }, [configLoading, refresh]);

  // Listen for deep link callback: gapfade://auth?token=xxx
  useEffect(() => {
    const handleUrl = async (event: {url: string}) => {
      const url = event.url;
      // Match token param from any redirect (Expo Go exp:// or standalone gapfade://)
      if (!url.includes('token=')) return;

      const match = url.match(/[?&]token=([^&]+)/);
      if (!match) return;

      const token = decodeURIComponent(match[1]);
      await SecureStore.setItemAsync(TOKEN_STORAGE_KEY, token);
      apiClient.setAuthToken(token);
      await refresh();
    };

    const sub = Linking.addEventListener('url', handleUrl);

    // Also check if app was opened via deep link (cold start)
    Linking.getInitialURL().then(url => {
      if (url) handleUrl({url});
    });

    return () => sub.remove();
  }, [apiClient, refresh]);

  const loginWithProvider = useCallback(
    async (provider: 'google' | 'github' | 'discord') => {
      // createURL returns the correct scheme for both Expo Go and standalone builds
      const redirectUrl = ExpoLinking.createURL('auth');
      const loginUrl = `${serverUrl}/api/auth/login/${provider}?mobile_redirect=${encodeURIComponent(redirectUrl)}`;

      // Use openAuthSessionAsync instead of Linking.openURL — it opens an
      // in-app browser tab (Chrome Custom Tab / SFSafariViewController) that
      // automatically intercepts the redirect back to our app scheme.
      const result = await WebBrowser.openAuthSessionAsync(loginUrl, redirectUrl);

      if (result.type === 'success' && result.url) {
        const match = result.url.match(/[?&]token=([^&]+)/);
        if (match) {
          const token = decodeURIComponent(match[1]);
          await SecureStore.setItemAsync(TOKEN_STORAGE_KEY, token);
          apiClient.setAuthToken(token);
          await refresh();
        }
      }
    },
    [serverUrl, apiClient, refresh],
  );

  const logout = useCallback(async () => {
    await SecureStore.deleteItemAsync(TOKEN_STORAGE_KEY);
    apiClient.setAuthToken('');
    setUser(null);
  }, [apiClient]);

  const value = useMemo(
    () => ({
      user,
      authEnabled,
      loading: loading || configLoading,
      refresh,
      loginWithProvider,
      logout,
      apiClient,
    }),
    [user, authEnabled, loading, configLoading, refresh, loginWithProvider, logout, apiClient],
  );

  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
}

export function useAuth(): AuthState {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be inside AuthProvider');
  return ctx;
}
