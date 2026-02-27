import React from 'react';
import {StyleSheet, Platform} from 'react-native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {Ionicons} from '@expo/vector-icons';
import {DashboardStack} from './stacks/DashboardStack';
import {ScannerStack} from './stacks/ScannerStack';
import {TradesStack} from './stacks/TradesStack';
import {TrackerStack} from './stacks/TrackerStack';
import {MoreStack} from './stacks/MoreStack';
import {colors} from '../theme';

const Tab = createBottomTabNavigator();

const TAB_ICONS: Record<string, keyof typeof Ionicons.glyphMap> = {
  DashboardTab: 'pie-chart',
  ScannerTab: 'scan',
  TradesTab: 'swap-horizontal',
  TrackerTab: 'pulse',
  MoreTab: 'ellipsis-horizontal',
};

const TAB_ICONS_OUTLINE: Record<string, keyof typeof Ionicons.glyphMap> = {
  DashboardTab: 'pie-chart-outline',
  ScannerTab: 'scan-outline',
  TradesTab: 'swap-horizontal-outline',
  TrackerTab: 'pulse-outline',
  MoreTab: 'ellipsis-horizontal',
};

export function TabNavigator() {
  return (
    <Tab.Navigator
      screenOptions={({route}) => ({
        headerShown: false,
        tabBarIcon: ({color, size, focused}) => (
          <Ionicons
            name={focused ? TAB_ICONS[route.name] : TAB_ICONS_OUTLINE[route.name]}
            size={22}
            color={color}
          />
        ),
        tabBarActiveTintColor: colors.cyan,
        tabBarInactiveTintColor: colors.textMuted,
        tabBarStyle: styles.tabBar,
        tabBarLabelStyle: styles.tabLabel,
      })}>
      <Tab.Screen
        name="DashboardTab"
        component={DashboardStack}
        options={{tabBarLabel: 'Dashboard'}}
      />
      <Tab.Screen
        name="ScannerTab"
        component={ScannerStack}
        options={{tabBarLabel: 'Scanner'}}
      />
      <Tab.Screen
        name="TradesTab"
        component={TradesStack}
        options={{tabBarLabel: 'Trades'}}
      />
      <Tab.Screen
        name="TrackerTab"
        component={TrackerStack}
        options={{tabBarLabel: 'Tracker'}}
      />
      <Tab.Screen
        name="MoreTab"
        component={MoreStack}
        options={{tabBarLabel: 'More'}}
      />
    </Tab.Navigator>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: colors.bgCard,
    borderTopColor: 'rgba(255, 255, 255, 0.06)',
    borderTopWidth: StyleSheet.hairlineWidth,
    height: Platform.OS === 'ios' ? 88 : 64,
    paddingTop: 6,
    paddingBottom: Platform.OS === 'ios' ? 28 : 8,
  },
  tabLabel: {
    fontSize: 10,
    fontWeight: '600',
    marginTop: 2,
  },
});
