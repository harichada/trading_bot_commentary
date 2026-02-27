import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {DashboardScreen} from '../../screens/DashboardScreen';
import {PositionDetailScreen} from '../../screens/PositionDetailScreen';
import {colors} from '../../theme';

export type DashboardStackParamList = {
  Dashboard: undefined;
  PositionDetail: {symbol: string};
};

const Stack = createNativeStackNavigator<DashboardStackParamList>();

export function DashboardStack() {
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
        name="Dashboard"
        component={DashboardScreen}
        options={{headerShown: false}}
      />
      <Stack.Screen
        name="PositionDetail"
        component={PositionDetailScreen}
        options={({route}) => ({title: route.params.symbol})}
      />
    </Stack.Navigator>
  );
}
