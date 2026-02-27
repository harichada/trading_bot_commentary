import React from 'react';
import {View, Text, ScrollView, StyleSheet} from 'react-native';
import {useRoute} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {RouteProp} from '@react-navigation/native';
import {useTradingState} from '../context/TradingStateContext';
import {GradientCard} from '../components/GradientCard';
import {DetailRow} from '../components/DetailRow';
import {ProgressRing} from '../components/ProgressRing';
import {EmptyState} from '../components/EmptyState';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {ScannerStackParamList} from '../navigation/stacks/ScannerStack';

type Route = RouteProp<ScannerStackParamList, 'CandidateDetail'>;

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);

export function CandidateDetailScreen() {
  const route = useRoute<Route>();
  const {symbol} = route.params;
  const {state} = useTradingState();
  const candidate = state.candidates.find(c => c.symbol === symbol);

  if (!candidate) {
    return (
      <View style={styles.container}>
        <EmptyState icon="search-outline" title={`${symbol} Not Found`} subtitle="Candidate may have been removed from scan results" />
      </View>
    );
  }

  const gapColor = candidate.gap_pct > 0 ? colors.green : colors.red;
  const dirColor = candidate.direction === 'short' ? colors.red : colors.green;
  const scoreColor = candidate.score >= 7 ? colors.green : candidate.score >= 4 ? colors.yellow : colors.red;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <View style={styles.header}>
          <View style={styles.headerLeft}>
            <Text style={styles.symbol}>{candidate.symbol}</Text>
            <View style={[styles.dirBadge, {backgroundColor: `${dirColor}20`}]}>
              <Text style={[styles.dirText, {color: dirColor}]}>
                {candidate.direction === 'short' ? 'SHORT' : 'LONG'}
              </Text>
            </View>
          </View>
          <ProgressRing
            progress={Math.min(candidate.score * 10, 100)}
            size={56}
            strokeWidth={4}
            color={scoreColor}
            label={candidate.score.toFixed(1)}
            showPercent={false}
          />
        </View>
      </Animated.View>

      {/* Gap Info */}
      <Animated.View entering={FadeInDown.delay(100).duration(300)}>
        <GradientCard accent={candidate.gap_pct > 0 ? 'green' : 'red'} style={styles.card}>
          <Text style={styles.sectionTitle}>Gap Info</Text>
          <DetailRow label="Gap %" value={`${candidate.gap_pct > 0 ? '+' : ''}${fmt(candidate.gap_pct, 2)}%`} valueColor={gapColor} mono />
          <DetailRow label="Prev Close" value={`$${fmt(candidate.prev_close)}`} mono />
          <DetailRow label="Premarket Price" value={`$${fmt(candidate.premarket_price)}`} mono />
          <DetailRow label="Direction" value={candidate.direction.toUpperCase()} valueColor={dirColor} />
        </GradientCard>
      </Animated.View>

      {/* Volume */}
      <Animated.View entering={FadeInDown.delay(150).duration(300)}>
        <GradientCard accent="cyan" style={styles.card}>
          <Text style={styles.sectionTitle}>Volume</Text>
          <DetailRow label="Avg Vol 20D" value={String(candidate.avg_vol_20d ?? 0)} mono />
          <DetailRow label="Vol Ratio" value={`${fmt(candidate.vol_ratio, 1)}x`} mono />
        </GradientCard>
      </Animated.View>

      {/* Borrowability */}
      <Animated.View entering={FadeInDown.delay(200).duration(300)}>
        <GradientCard accent="none" style={styles.card}>
          <Text style={styles.sectionTitle}>Borrowability</Text>
          <DetailRow label="Shortable" value={candidate.shortable ? 'Yes' : 'No'} valueColor={candidate.shortable ? colors.green : colors.red} />
          <DetailRow label="Easy to Borrow" value={candidate.easy_to_borrow ? 'Yes' : 'No'} valueColor={candidate.easy_to_borrow ? colors.green : colors.yellow} />
        </GradientCard>
      </Animated.View>

      {/* Catalyst */}
      <Animated.View entering={FadeInDown.delay(250).duration(300)}>
        <GradientCard accent={candidate.catalyst ? 'purple' : 'none'} style={styles.card}>
          <Text style={styles.sectionTitle}>Catalyst</Text>
          <DetailRow label="Type" value={candidate.catalyst || 'None'} />
          {candidate.catalyst_detail ? (
            <Text style={styles.catalystDetail}>{candidate.catalyst_detail}</Text>
          ) : null}
        </GradientCard>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingBottom: spacing.xxxl},
  header: {flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: spacing.lg},
  headerLeft: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm},
  symbol: {...typography.h1, color: colors.textPrimary},
  dirBadge: {paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm},
  dirText: {...typography.label, fontSize: 11},
  card: {marginBottom: spacing.md},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.sm},
  catalystDetail: {...typography.body, color: colors.textSecondary, marginTop: spacing.sm, lineHeight: 20},
});
