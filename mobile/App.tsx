import React from 'react';
import {GestureHandlerRootView} from 'react-native-gesture-handler';
import {StyleSheet} from 'react-native';
import {ServerConfigProvider} from './src/context/ServerConfigContext';
import {AuthProvider} from './src/context/AuthContext';
import {WebSocketProvider} from './src/context/WebSocketContext';
import {TradingStateProvider} from './src/context/TradingStateContext';
import {RootNavigator} from './src/navigation/RootNavigator';
import {ToastProvider} from './src/components/Toast';

export default function App() {
  return (
    <GestureHandlerRootView style={styles.root}>
      <ServerConfigProvider>
        <AuthProvider>
          <WebSocketProvider>
            <TradingStateProvider>
              <ToastProvider>
                <RootNavigator />
              </ToastProvider>
            </TradingStateProvider>
          </WebSocketProvider>
        </AuthProvider>
      </ServerConfigProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
});
