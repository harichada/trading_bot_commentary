import React from 'react';
import {View, Text, ScrollView, StyleSheet} from 'react-native';
import {useRoute} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {RouteProp} from '@react-navigation/native';
import {useTradingState} from '../context/TradingStateContext';
import {GradientCard} from '../components/GradientCard';
import {DetailRow} from '../components/DetailRow';
import {EmptyState} from '../components/EmptyState';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TradesStackParamList} from '../navigation/stacks/TradesStack';

type Route = RouteProp<TradesStackParamList, 'TradeDetail'>;

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);

const exitReasonLabels: Record<string, string> = {
  stop: 'Stop Loss',
  partial: 'Partial Target',
  full_target: 'Full Target',
  time_exit: 'Time Exit',
  eod: 'End of Day',
  manual: 'Manual Close',
};

export function TradeDetailScreen() {
  const route = useRoute<Route>();
  const {index} = route.params;
  const {state} = useTradingState();
  const trade = state.today_trades[index];

  if (!trade) {
    return (
      <View style={styles.container}>
        <EmptyState icon="receipt-outline" title="Trade Not Found" subtitle="This trade may no longer be available" />
      </View>
    );
  }

  const isWin = (trade.pnl ?? 0) >= 0;
  const pnlColor = isWin ? colors.green : colors.red;
  const sideColor = trade.side === 'short' ? colors.red : colors.green;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <View style={styles.header}>
          <View>
            <Text style={styles.symbol}>{trade.symbol}</Text>
            <View style={[styles.sideBadge, {backgroundColor: `${sideColor}20`}]}>
              <Text style={[styles.sideText, {color: sideColor}]}>{trade.side?.toUpperCase()}</Text>
            </View>
          </View>
          <View style={{alignItems: 'flex-end'}}>
            <Text style={[styles.pnlValue, {color: pnlColor}]}>
              {isWin ? '+' : ''}${fmt(trade.pnl)}
            </Text>
            <Text style={[styles.pnlPct, {color: pnlColor}]}>
              {isWin ? '+' : ''}{fmt(trade.pnl_pct)}%
            </Text>
          </View>
        </View>
      </Animated.View>

      {/* Execution */}
      <Animated.View entering={FadeInDown.delay(100).duration(300)}>
        <GradientCard accent={isWin ? 'green' : 'red'} style={styles.card}>
          <Text style={styles.sectionTitle}>Execution</Text>
          <DetailRow label="Entry Price" value={`$${fmt(trade.entry_price)}`} mono />
          <DetailRow label="Exit Price" value={`$${fmt(trade.exit_price)}`} mono />
          <DetailRow label="Shares" value={String(trade.shares ?? 0)} mono />
          <DetailRow label="Exit Reason" value={exitReasonLabels[trade.exit_reason] || trade.exit_reason} />
          <DetailRow label="Holding Time" value={`${fmt(trade.holding_minutes, 0)} min`} mono />
        </GradientCard>
      </Animated.View>

      {/* Timing */}
      <Animated.View entering={FadeInDown.delay(150).duration(300)}>
        <GradientCard accent="none" style={styles.card}>
          <Text style={styles.sectionTitle}>Timing</Text>
          <DetailRow label="Entry Time" value={trade.entry_time || '-'} />
          <DetailRow label="Exit Time" value={trade.exit_time || '-'} />
        </GradientCard>
      </Animated.View>

      {/* Scan Data */}
      <Animated.View entering={FadeInDown.delay(200).duration(300)}>
        <GradientCard accent="cyan" style={styles.card}>
          <Text style={styles.sectionTitle}>Scan Data</Text>
          <DetailRow label="Gap %" value={`${fmt(trade.gap_pct, 1)}%`} mono />
          <DetailRow label="Vol Ratio" value={`${fmt(trade.vol_ratio, 1)}x`} mono />
          <DetailRow label="Score" value={fmt(trade.score, 1)} mono />
          <DetailRow label="Catalyst" value={trade.catalyst || 'None'} />
        </GradientCard>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingBottom: spacing.xxxl},
  header: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: spacing.lg},
  symbol: {...typography.h1, color: colors.textPrimary},
  sideBadge: {alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm, marginTop: spacing.xs},
  sideText: {...typography.label, fontSize: 11},
  pnlValue: {...typography.heroMedium},
  pnlPct: {...typography.mono, fontSize: 16, marginTop: 2},
  card: {marginBottom: spacing.md},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.sm},
});
