import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {TradeHistoryScreen} from '../../screens/TradeHistoryScreen';
import {TradeDetailScreen} from '../../screens/TradeDetailScreen';
import {colors} from '../../theme';

export type TradesStackParamList = {
  TradeHistory: undefined;
  TradeDetail: {index: number};
};

const Stack = createNativeStackNavigator<TradesStackParamList>();

export function TradesStack() {
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
        name="TradeHistory"
        component={TradeHistoryScreen}
        options={{headerShown: false}}
      />
      <Stack.Screen
        name="TradeDetail"
        component={TradeDetailScreen}
        options={{title: 'Trade Detail'}}
      />
    </Stack.Navigator>
  );
}
