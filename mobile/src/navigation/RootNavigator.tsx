import React from 'react';
import {ActivityIndicator, View, StyleSheet} from 'react-native';
import {StatusBar} from 'expo-status-bar';
import {NavigationContainer, DefaultTheme} from '@react-navigation/native';
import {useServerConfig} from '../context/ServerConfigContext';
import {useAuth} from '../context/AuthContext';
import {ServerSetupScreen} from '../screens/ServerSetupScreen';
import {LoginScreen} from '../screens/LoginScreen';
import {TabNavigator} from './TabNavigator';
import {colors} from '../theme';

const DarkNavTheme = {
  ...DefaultTheme,
  dark: true,
  colors: {
    ...DefaultTheme.colors,
    primary: colors.cyan,
    background: colors.bg,
    card: colors.bgCard,
    text: colors.textPrimary,
    border: colors.border,
    notification: colors.red,
  },
};

export function RootNavigator() {
  const {serverUrl, loading: configLoading, setServerUrl, setApiKey} = useServerConfig();
  const {user, authEnabled, loading: authLoading} = useAuth();

  if (configLoading || authLoading) {
    return (
      <View style={styles.loader}>
        <ActivityIndicator size="large" color={colors.cyan} />
      </View>
    );
  }

  if (!serverUrl) {
    return (
      <ServerSetupScreen
        onConnect={async (url, apiKey) => {
          await setServerUrl(url);
          await setApiKey(apiKey);
        }}
      />
    );
  }

  if (authEnabled && !user) {
    return <LoginScreen />;
  }

  return (
    <NavigationContainer theme={DarkNavTheme}>
      <StatusBar style="light" />
      <TabNavigator />
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  loader: {
    flex: 1,
    backgroundColor: colors.bg,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
