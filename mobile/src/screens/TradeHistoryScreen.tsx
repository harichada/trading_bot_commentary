import React, {useCallback, useState, useMemo} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ScrollView,
} from 'react-native';
import {useNavigation} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import * as Haptics from 'expo-haptics';
import {useTradingState} from '../context/TradingStateContext';
import {FilterChips} from '../components/FilterChips';
import {GradientCard} from '../components/GradientCard';
import {EmptyState} from '../components/EmptyState';
import {SectionHeader} from '../components/SectionHeader';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TradeRecord} from '../types/trading';
import type {TradesStackParamList} from '../navigation/stacks/TradesStack';

type Nav = NativeStackNavigationProp<TradesStackParamList, 'TradeHistory'>;

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);
const FILTERS = ['All', 'Wins', 'Losses', 'Stops', 'Targets', 'EOD'];

function TradeRow({
  trade,
  index,
  onPress,
}: {
  trade: TradeRecord;
  index: number;
  onPress: () => void;
}) {
  const isWin = trade.pnl >= 0;
  const pnlColor = isWin ? colors.green : colors.red;
  const sideColor = trade.side === 'short' ? colors.red : colors.green;
  const accent = isWin ? 'green' as const : 'red' as const;

  return (
    <Animated.View entering={FadeInDown.delay((index % 10) * 50).duration(250)}>
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        <GradientCard accent={accent} style={rowStyles.card} padding={spacing.md}>
          <View style={rowStyles.topRow}>
            <View style={rowStyles.symbolRow}>
              <Text style={rowStyles.symbol}>{trade.symbol}</Text>
              <View style={[rowStyles.sideBadge, {backgroundColor: `${sideColor}20`}]}>
                <Text style={[rowStyles.sideText, {color: sideColor}]}>
                  {trade.side.toUpperCase()}
                </Text>
              </View>
              <Text style={rowStyles.exitReason}>{trade.exit_reason}</Text>
            </View>
            <View style={{alignItems: 'flex-end'}}>
              <Text style={[rowStyles.pnl, {color: pnlColor}]}>
                {isWin ? '+' : ''}${trade.pnl.toFixed(2)}
              </Text>
              <Text style={[rowStyles.pnlPct, {color: pnlColor}]}>
                {isWin ? '+' : ''}{trade.pnl_pct.toFixed(2)}%
              </Text>
            </View>
          </View>
        </GradientCard>
      </TouchableOpacity>
    </Animated.View>
  );
}

const rowStyles = StyleSheet.create({
  card: {marginBottom: spacing.sm},
  topRow: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center'},
  symbolRow: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm},
  symbol: {...typography.h3, color: colors.textPrimary},
  sideBadge: {paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm},
  sideText: {...typography.label, fontSize: 10},
  exitReason: {...typography.caption, color: colors.textMuted},
  pnl: {...typography.monoLarge},
  pnlPct: {...typography.monoSmall, marginTop: 2},
});

// Mini P&L Distribution Bar
function PnlBar({trades}: {trades: TradeRecord[]}) {
  if (trades.length === 0) return null;
  const maxPnl = Math.max(...trades.map(t => Math.abs(t.pnl)), 1);

  return (
    <View style={barStyles.container}>
      {trades.slice(-30).map((t, i) => {
        const isWin = t.pnl >= 0;
        const height = Math.max(4, (Math.abs(t.pnl) / maxPnl) * 32);
        return (
          <View
            key={`${t.symbol}-${i}`}
            style={[
              barStyles.bar,
              {
                height,
                backgroundColor: isWin ? colors.green : colors.red,
              },
            ]}
          />
        );
      })}
    </View>
  );
}

const barStyles = StyleSheet.create({
  container: {flexDirection: 'row', alignItems: 'flex-end', gap: 2, height: 36, paddingVertical: 2},
  bar: {width: 6, borderRadius: 2},
});

export function TradeHistoryScreen() {
  const navigation = useNavigation<Nav>();
  const {state, refreshState} = useTradingState();
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState('All');

  const trades = state.today_trades;

  const filtered = useMemo(() => {
    return trades.filter(t => {
      switch (filter) {
        case 'Wins': return t.pnl >= 0;
        case 'Losses': return t.pnl < 0;
        case 'Stops': return t.exit_reason === 'stop';
        case 'Targets': return t.exit_reason === 'full_target' || t.exit_reason === 'partial';
        case 'EOD': return t.exit_reason === 'eod' || t.exit_reason === 'time_exit';
        default: return true;
      }
    });
  }, [trades, filter]);

  const totalPnl = trades.reduce((sum, t) => sum + t.pnl, 0);
  const wins = trades.filter(t => t.pnl >= 0).length;
  const winRate = trades.length > 0 ? (wins / trades.length) * 100 : 0;
  const pnlColor = totalPnl >= 0 ? colors.green : colors.red;

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refreshState();
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setRefreshing(false);
  }, [refreshState]);

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.cyan} colors={[colors.cyan]} />
        }
      >
        <Text style={styles.title}>Trades</Text>

        {/* Summary Card */}
        <Animated.View entering={FadeInDown.duration(300)}>
          <GradientCard accent={totalPnl >= 0 ? 'green' : 'red'} style={styles.summaryCard}>
            <View style={styles.summaryRow}>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>TOTAL</Text>
                <Text style={styles.summaryValue}>{trades.length}</Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>P&L</Text>
                <Text style={[styles.summaryValueMono, {color: pnlColor}]}>
                  {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
                </Text>
              </View>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryLabel}>WIN RATE</Text>
                <Text style={styles.summaryValueMono}>{winRate.toFixed(1)}%</Text>
              </View>
            </View>
            <PnlBar trades={trades} />
          </GradientCard>
        </Animated.View>

        {/* Filter Chips */}
        <FilterChips options={FILTERS} selected={filter} onSelect={setFilter} />

        {/* Trade List */}
        {filtered.length === 0 ? (
          <EmptyState
            icon="receipt-outline"
            title="No Trades"
            subtitle={filter === 'All' ? 'No trades recorded today' : `No ${filter.toLowerCase()} trades`}
          />
        ) : (
          <View style={styles.list}>
            {filtered.map((trade, index) => (
              <TradeRow
                key={`${trade.symbol}-${trade.exit_time}-${index}`}
                trade={trade}
                index={index}
                onPress={() => navigation.navigate('TradeDetail', {index: trades.indexOf(trade)})}
              />
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  scrollContent: {paddingTop: spacing.xl + 40, paddingBottom: spacing.xxxl},
  title: {...typography.h1, color: colors.textPrimary, paddingHorizontal: spacing.lg, marginBottom: spacing.md},
  summaryCard: {marginHorizontal: spacing.lg, marginBottom: spacing.sm},
  summaryRow: {flexDirection: 'row', marginBottom: spacing.sm},
  summaryItem: {flex: 1, alignItems: 'center'},
  summaryLabel: {...typography.label, color: colors.textMuted, marginBottom: spacing.xs},
  summaryValue: {...typography.h3, color: colors.textPrimary},
  summaryValueMono: {...typography.monoLarge, color: colors.textPrimary},
  list: {paddingHorizontal: spacing.lg, paddingTop: spacing.sm},
});
