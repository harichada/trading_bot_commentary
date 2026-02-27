import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {TrackerScreen} from '../../screens/TrackerScreen';
import {colors} from '../../theme';

export type TrackerStackParamList = {
  Tracker: undefined;
};

const Stack = createNativeStackNavigator<TrackerStackParamList>();

export function TrackerStack() {
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
        name="Tracker"
        component={TrackerScreen}
        options={{headerShown: false}}
      />
    </Stack.Navigator>
  );
}
