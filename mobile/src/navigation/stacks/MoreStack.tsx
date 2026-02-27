import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {MoreMenuScreen} from '../../screens/MoreMenuScreen';
import {ConfigScreen} from '../../screens/ConfigScreen';
import {BacktestScreen} from '../../screens/BacktestScreen';
import {BacktestResultScreen} from '../../screens/BacktestResultScreen';
import {DatabaseScreen} from '../../screens/DatabaseScreen';
import {ChatScreen} from '../../screens/ChatScreen';
import {GuideScreen} from '../../screens/GuideScreen';
import {SettingsScreen} from '../../screens/SettingsScreen';
import {AccountScreen} from '../../screens/AccountScreen';
import {colors} from '../../theme';

export type MoreStackParamList = {
  MoreMenu: undefined;
  Config: undefined;
  Backtest: undefined;
  BacktestResult: undefined;
  Database: undefined;
  Chat: undefined;
  Guide: undefined;
  Settings: undefined;
  Account: undefined;
};

const Stack = createNativeStackNavigator<MoreStackParamList>();

export function MoreStack() {
  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {backgroundColor: colors.bg},
        headerTintColor: colors.cyan,
        headerTitleStyle: {color: colors.textPrimary, fontWeight: '600'},
        headerShadowVisible: false,
        animation: 'slide_from_right',
        animationDuration: 200,
      }}>
      <Stack.Screen
        name="MoreMenu"
        component={MoreMenuScreen}
        options={{headerShown: false}}
      />
      <Stack.Screen name="Config" component={ConfigScreen} options={{title: 'Configuration'}} />
      <Stack.Screen name="Backtest" component={BacktestScreen} options={{title: 'Backtest'}} />
      <Stack.Screen
        name="BacktestResult"
        component={BacktestResultScreen}
        options={{title: 'Backtest Results'}}
      />
      <Stack.Screen name="Database" component={DatabaseScreen} options={{title: 'Database'}} />
      <Stack.Screen name="Chat" component={ChatScreen} options={{title: 'Rudra Chat'}} />
      <Stack.Screen name="Guide" component={GuideScreen} options={{title: 'Guide'}} />
      <Stack.Screen name="Settings" component={SettingsScreen} options={{title: 'Settings'}} />
      <Stack.Screen name="Account" component={AccountScreen} options={{title: 'Account'}} />
    </Stack.Navigator>
  );
}
