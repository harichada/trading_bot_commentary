import React from 'react';
import {View, Text, TouchableOpacity, StyleSheet} from 'react-native';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {GapCandidate} from '../types/trading';

interface CandidateRowProps {
  candidate: GapCandidate;
  onPress: () => void;
}

export const CandidateRow: React.FC<CandidateRowProps> = ({
  candidate,
  onPress,
}) => {
  const isShort = candidate.direction === 'short';
  const directionColor = isShort ? colors.red : colors.green;
  const directionLabel = isShort ? 'SHORT' : 'LONG';
  const gapSign = candidate.gap_pct >= 0 ? '+' : '';

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onPress}
      activeOpacity={0.7}>
      {/* Left: Symbol + Direction */}
      <View style={styles.leftSection}>
        <View style={styles.symbolRow}>
          <Text style={styles.symbol}>{candidate.symbol}</Text>
          <View
            style={[
              styles.directionBadge,
              {backgroundColor: `${directionColor}20`},
            ]}>
            <Text style={[styles.directionText, {color: directionColor}]}>
              {directionLabel}
            </Text>
          </View>
        </View>
        {candidate.catalyst ? (
          <Text style={styles.catalyst} numberOfLines={1}>
            {candidate.catalyst}
          </Text>
        ) : null}
      </View>

      {/* Right: Metrics */}
      <View style={styles.rightSection}>
        <View style={styles.metricsRow}>
          <Text style={styles.gapPct}>
            {gapSign}
            {candidate.gap_pct.toFixed(1)}%
          </Text>
          <Text style={styles.score}>{candidate.score.toFixed(1)}</Text>
        </View>
        <Text style={styles.volRatio}>
          Vol: {candidate.vol_ratio.toFixed(1)}x
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
  directionBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  directionText: {
    ...typography.label,
  },
  catalyst: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 2,
  },
  rightSection: {
    alignItems: 'flex-end',
  },
  metricsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  gapPct: {
    ...typography.mono,
    color: colors.cyan,
  },
  score: {
    ...typography.monoSmall,
    color: colors.textSecondary,
    backgroundColor: colors.bgCardHover,
    paddingHorizontal: spacing.xs,
    paddingVertical: 1,
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
  },
  volRatio: {
    ...typography.caption,
    color: colors.textMuted,
    marginTop: 2,
  },
});

export default CandidateRow;
