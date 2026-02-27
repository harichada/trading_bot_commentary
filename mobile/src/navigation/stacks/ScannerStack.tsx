import React from 'react';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {ScannerScreen} from '../../screens/ScannerScreen';
import {CandidateDetailScreen} from '../../screens/CandidateDetailScreen';
import {colors} from '../../theme';

export type ScannerStackParamList = {
  Scanner: undefined;
  CandidateDetail: {symbol: string};
};

const Stack = createNativeStackNavigator<ScannerStackParamList>();

export function ScannerStack() {
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
        name="Scanner"
        component={ScannerScreen}
        options={{headerShown: false}}
      />
      <Stack.Screen
        name="CandidateDetail"
        component={CandidateDetailScreen}
        options={({route}) => ({title: route.params.symbol})}
      />
    </Stack.Navigator>
  );
}
