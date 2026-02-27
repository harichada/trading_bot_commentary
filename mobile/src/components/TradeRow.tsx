import React from 'react';
import {View, Text, TouchableOpacity, StyleSheet} from 'react-native';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TradeRecord} from '../types/trading';

interface TradeRowProps {
  trade: TradeRecord;
  onPress: () => void;
}

const exitReasonLabels: Record<TradeRecord['exit_reason'], string> = {
  stop: 'Stopped',
  partial: 'Partial',
  full_target: 'Target',
  time_exit: 'Time',
  eod: 'EOD',
  manual: 'Manual',
};

export const TradeRow: React.FC<TradeRowProps> = ({trade, onPress}) => {
  const isShort = trade.side === 'short';
  const sideColor = isShort ? colors.red : colors.green;
  const sideLabel = isShort ? 'SHORT' : 'LONG';

  const pnlColor = trade.pnl >= 0 ? colors.green : colors.red;
  const pnlSign = trade.pnl >= 0 ? '+' : '';

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onPress}
      activeOpacity={0.7}>
      {/* Left: Symbol + Side + Exit Reason */}
      <View style={styles.leftSection}>
        <View style={styles.symbolRow}>
          <Text style={styles.symbol}>{trade.symbol}</Text>
          <View
            style={[
              styles.sideBadge,
              {backgroundColor: `${sideColor}20`},
            ]}>
            <Text style={[styles.sideText, {color: sideColor}]}>
              {sideLabel}
            </Text>
          </View>
        </View>
        <View style={styles.metaRow}>
          <Text style={styles.exitReason}>
            {exitReasonLabels[trade.exit_reason]}
          </Text>
          <Text style={styles.separator}>{'\u00B7'}</Text>
          <Text style={styles.holdingTime}>{trade.holding_minutes}m</Text>
        </View>
      </View>

      {/* Right: P&L */}
      <View style={styles.rightSection}>
        <Text style={[styles.pnl, {color: pnlColor}]}>
          {pnlSign}${trade.pnl.toFixed(2)}
        </Text>
        <Text style={[styles.pnlPct, {color: pnlColor}]}>
          {pnlSign}
          {trade.pnl_pct.toFixed(2)}%
        </Text>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: colors.bgCard,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.sm,
  },
  leftSection: {
    flex: 1,
    marginRight: spacing.md,
  },
  symbolRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  symbol: {
    ...typography.h3,
    color: colors.textPrimary,
  },
  sideBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  sideText: {
    ...typography.label,
  },
  metaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
    gap: spacing.xs,
  },
  exitReason: {
    ...typography.caption,
    color: colors.textMuted,
  },
  separator: {
    ...typography.caption,
    color: colors.textMuted,
  },
  holdingTime: {
    ...typography.caption,
    color: colors.textMuted,
  },
  rightSection: {
    alignItems: 'flex-end',
  },
  pnl: {
    ...typography.mono,
  },
  pnlPct: {
    ...typography.monoSmall,
    marginTop: 2,
  },
});

export default TradeRow;
