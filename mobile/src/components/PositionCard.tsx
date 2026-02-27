import React from 'react';
import {View, Text, TouchableOpacity, StyleSheet} from 'react-native';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {GapPosition} from '../types/trading';

interface PositionCardProps {
  position: GapPosition;
  onClose: (symbol: string) => void;
  onAdjustStop: (symbol: string) => void;
}

export const PositionCard: React.FC<PositionCardProps> = ({
  position,
  onClose,
  onAdjustStop,
}) => {
  const isShort = position.direction === 'short';
  const directionColor = isShort ? colors.red : colors.green;
  const directionLabel = isShort ? 'SHORT' : 'LONG';

  // Calculate unrealized P&L using entry_price and high_water_price
  const unrealizedPnl = isShort
    ? ((position.entry_price - position.high_water_price) /
        position.entry_price) *
      100
    : ((position.high_water_price - position.entry_price) /
        position.entry_price) *
      100;

  const pnlColor = unrealizedPnl >= 0 ? colors.green : colors.red;
  const pnlSign = unrealizedPnl >= 0 ? '+' : '';

  return (
    <View style={styles.container}>
      {/* Header: Symbol + Direction */}
      <View style={styles.header}>
        <View style={styles.symbolRow}>
          <Text style={styles.symbol}>{position.symbol}</Text>
          <View
            style={[
              styles.directionBadge,
              {backgroundColor: `${directionColor}20`},
            ]}>
            <Text style={[styles.directionText, {color: directionColor}]}>
              {directionLabel}
            </Text>
          </View>
          {position.partial_filled && (
            <View style={styles.partialBadge}>
              <Text style={styles.partialText}>PARTIAL</Text>
            </View>
          )}
        </View>
        <Text style={[styles.pnl, {color: pnlColor}]}>
          {pnlSign}
          {unrealizedPnl.toFixed(2)}%
        </Text>
      </View>

      {/* Details Row */}
      <View style={styles.detailsRow}>
        <View style={styles.detail}>
          <Text style={styles.detailLabel}>Entry</Text>
          <Text style={styles.detailValue}>
            ${position.entry_price.toFixed(2)}
          </Text>
        </View>
        <View style={styles.detail}>
          <Text style={styles.detailLabel}>Stop</Text>
          <Text style={styles.detailValue}>
            ${position.stop_price.toFixed(2)}
          </Text>
        </View>
        <View style={styles.detail}>
          <Text style={styles.detailLabel}>HWM</Text>
          <Text style={styles.detailValue}>
            ${position.high_water_price.toFixed(2)}
          </Text>
        </View>
        <View style={styles.detail}>
          <Text style={styles.detailLabel}>Shares</Text>
          <Text style={styles.detailValue}>
            {position.remaining_shares}/{position.shares}
          </Text>
        </View>
      </View>

      {/* Action Buttons */}
      <View style={styles.actions}>
        <TouchableOpacity
          style={styles.closeButton}
          onPress={() => onClose(position.symbol)}
          activeOpacity={0.7}>
          <Text style={styles.closeButtonText}>Close</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.adjustButton}
          onPress={() => onAdjustStop(position.symbol)}
          activeOpacity={0.7}>
          <Text style={styles.adjustButtonText}>Adj Stop</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.bgCard,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    marginBottom: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
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
  directionBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  directionText: {
    ...typography.label,
  },
  partialBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
    backgroundColor: colors.yellowDim,
  },
  partialText: {
    ...typography.label,
    color: colors.yellow,
  },
  pnl: {
    ...typography.monoLarge,
  },
  detailsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.md,
    paddingTop: spacing.sm,
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  detail: {
    alignItems: 'center',
  },
  detailLabel: {
    ...typography.caption,
    color: colors.textMuted,
    marginBottom: 2,
  },
  detailValue: {
    ...typography.monoSmall,
    color: colors.textSecondary,
  },
  actions: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  closeButton: {
    flex: 1,
    backgroundColor: colors.redDim,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: `${colors.red}30`,
  },
  closeButtonText: {
    ...typography.bodyMedium,
    color: colors.red,
  },
  adjustButton: {
    flex: 1,
    backgroundColor: colors.cyanDim,
    borderRadius: borderRadius.md,
    paddingVertical: spacing.sm,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: `${colors.cyan}30`,
  },
  adjustButtonText: {
    ...typography.bodyMedium,
    color: colors.cyan,
  },
});

export default PositionCard;
