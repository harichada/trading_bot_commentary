import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import {AnimatedNumber} from './AnimatedNumber';
import {PulsingDot} from './PulsingDot';
import {StatusBadge} from './StatusBadge';
import {colors, typography, spacing} from '../theme';
import type {TraderStatus} from '../types/trading';

interface HeroSectionProps {
  equity: number;
  dailyPnl: number;
  dailyPnlPct: number;
  status: TraderStatus;
  connected: boolean;
}

export function HeroSection({
  equity,
  dailyPnl,
  dailyPnlPct,
  status,
  connected,
}: HeroSectionProps) {
  const pnlColor = dailyPnl >= 0 ? colors.green : colors.red;
  const pnlSign = dailyPnl >= 0 ? '+' : '';

  return (
    <View style={styles.container}>
      {/* Top bar: status + connection */}
      <View style={styles.topRow}>
        <View style={styles.titleRow}>
          <Text style={styles.appTitle}>Gap Fade</Text>
          <StatusBadge status={status} />
        </View>
        <View style={styles.connRow}>
          <PulsingDot
            color={connected ? colors.green : colors.red}
            size={7}
            active={connected}
          />
          <Text style={[styles.connText, {color: connected ? colors.green : colors.red}]}>
            {connected ? 'Live' : 'Offline'}
          </Text>
        </View>
      </View>

      {/* Big equity number */}
      <AnimatedNumber
        value={equity}
        prefix="$"
        decimals={2}
        style={styles.equity}
        duration={800}
      />

      {/* Daily P&L */}
      <View style={styles.pnlRow}>
        <Text style={[styles.pnlText, {color: pnlColor}]}>
          {pnlSign}${Math.abs(dailyPnl).toFixed(2)}
        </Text>
        <Text style={[styles.pnlPct, {color: pnlColor}]}>
          ({pnlSign}{dailyPnlPct.toFixed(2)}%)
        </Text>
        <Text style={styles.todayLabel}>Today</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingBottom: spacing.sm,
  },
  topRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.md,
  },
  appTitle: {
    ...typography.h2,
    color: colors.textPrimary,
  },
  connRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs,
  },
  connText: {
    ...typography.caption,
    fontWeight: '600',
  },
  equity: {
    ...typography.heroLarge,
    color: colors.textPrimary,
    marginBottom: spacing.xs,
  },
  pnlRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  pnlText: {
    ...typography.monoLarge,
  },
  pnlPct: {
    ...typography.mono,
  },
  todayLabel: {
    ...typography.caption,
    color: colors.textMuted,
  },
});
