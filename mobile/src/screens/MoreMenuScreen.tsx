import React from 'react';
import {View, Text, TouchableOpacity, StyleSheet, ScrollView} from 'react-native';
import {useNavigation} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import Constants from 'expo-constants';
import {useTradingState} from '../context/TradingStateContext';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {MoreStackParamList} from '../navigation/stacks/MoreStack';

const APP_VERSION = Constants.expoConfig?.version ?? '1.0.0';

type Nav = NativeStackNavigationProp<MoreStackParamList, 'MoreMenu'>;

interface MenuItem {
  label: string;
  subtitle: string;
  route: keyof MoreStackParamList;
  icon: keyof typeof Ionicons.glyphMap;
  iconColor?: string;
}

const MENU_ITEMS: MenuItem[] = [
  {label: 'Configuration', subtitle: 'Strategy parameters', route: 'Config', icon: 'settings', iconColor: colors.cyan},
  {label: 'Backtest', subtitle: 'Run & review backtests', route: 'Backtest', icon: 'bar-chart', iconColor: colors.purple},
  {label: 'Database', subtitle: 'OHLCV data management', route: 'Database', icon: 'server', iconColor: colors.green},
  {label: 'Rudra Chat', subtitle: 'LLM supervisor chat', route: 'Chat', icon: 'chatbubbles', iconColor: colors.yellow},
  {label: 'Guide', subtitle: 'Strategy documentation', route: 'Guide', icon: 'book', iconColor: colors.orange},
  {label: 'Account', subtitle: 'Alpaca account info', route: 'Account', icon: 'person', iconColor: colors.cyan},
  {label: 'Settings', subtitle: 'App & server settings', route: 'Settings', icon: 'cog', iconColor: colors.textSecondary},
];

function MenuRow({
  item,
  onPress,
  badge,
  index,
}: {
  item: MenuItem;
  onPress: () => void;
  badge?: string;
  index: number;
}) {
  return (
    <Animated.View entering={FadeInDown.delay(index * 40).duration(200)}>
      <TouchableOpacity style={styles.row} onPress={onPress} activeOpacity={0.7}>
        <View style={[styles.iconContainer, {backgroundColor: `${item.iconColor}15`}]}>
          <Ionicons name={item.icon} size={20} color={item.iconColor} />
        </View>
        <View style={styles.rowContent}>
          <View style={styles.rowLabelRow}>
            <Text style={styles.rowLabel}>{item.label}</Text>
            {badge && (
              <View style={styles.badge}>
                <Text style={styles.badgeText}>{badge}</Text>
              </View>
            )}
          </View>
          <Text style={styles.rowSubtitle}>{item.subtitle}</Text>
        </View>
        <Ionicons name="chevron-forward" size={18} color={colors.textMuted} />
      </TouchableOpacity>
    </Animated.View>
  );
}

export function MoreMenuScreen() {
  const navigation = useNavigation<Nav>();
  const {state} = useTradingState();

  const getBadge = (route: string): string | undefined => {
    if (route === 'Backtest' && state.backtest?.running) return 'Running';
    if (route === 'Chat' && state.llm?.available) return 'Online';
    return undefined;
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>More</Text>

      <View style={styles.card}>
        {MENU_ITEMS.map((item, index) => (
          <React.Fragment key={item.route}>
            <MenuRow
              item={item}
              onPress={() => navigation.navigate(item.route)}
              badge={getBadge(item.route)}
              index={index}
            />
            {index < MENU_ITEMS.length - 1 && <View style={styles.separator} />}
          </React.Fragment>
        ))}
      </View>

      <Text style={styles.version}>Gap Fade Mobile v{APP_VERSION}</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingTop: spacing.xl + 40},
  title: {...typography.h1, color: colors.textPrimary, marginBottom: spacing.lg},
  card: {
    backgroundColor: colors.bgCard, borderRadius: borderRadius.xl,
    borderWidth: 1, borderColor: colors.border, overflow: 'hidden',
  },
  row: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: spacing.lg, paddingVertical: spacing.md + 2, gap: spacing.md,
  },
  iconContainer: {
    width: 36, height: 36, borderRadius: 10,
    alignItems: 'center', justifyContent: 'center',
  },
  rowContent: {flex: 1},
  rowLabelRow: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm},
  rowLabel: {...typography.bodyMedium, color: colors.textPrimary},
  rowSubtitle: {...typography.caption, color: colors.textMuted, marginTop: 2},
  badge: {
    backgroundColor: 'rgba(0, 212, 255, 0.12)', paddingHorizontal: spacing.sm,
    paddingVertical: 1, borderRadius: borderRadius.round,
  },
  badgeText: {...typography.label, color: colors.cyan, fontSize: 9},
  separator: {height: 1, backgroundColor: colors.border, marginLeft: spacing.lg + 36 + spacing.md},
  version: {...typography.caption, color: colors.textMuted, textAlign: 'center', marginTop: spacing.xl},
});
