import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import {Ionicons} from '@expo/vector-icons';
import {GradientCard} from './GradientCard';
import {colors, typography, spacing} from '../theme';

interface MetricCardProps {
  label: string;
  value: string;
  color?: string;
  mono?: boolean;
  trend?: 'up' | 'down' | 'neutral';
  accent?: 'green' | 'red' | 'cyan' | 'purple' | 'none';
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  color,
  mono = false,
  trend,
  accent = 'none',
}) => {
  const trendIcon = trend === 'up' ? 'trending-up' : trend === 'down' ? 'trending-down' : null;
  const trendColor = trend === 'up' ? colors.green : trend === 'down' ? colors.red : colors.textMuted;

  return (
    <GradientCard accent={accent} padding={spacing.md} style={styles.container}>
      <View style={styles.labelRow}>
        <Text style={styles.label}>{label}</Text>
        {trendIcon && (
          <Ionicons name={trendIcon} size={12} color={trendColor} />
        )}
      </View>
      <Text
        style={[
          mono ? styles.valueMono : styles.value,
          color ? {color} : null,
        ]}
        numberOfLines={1}
        adjustsFontSizeToFit>
        {value}
      </Text>
    </GradientCard>
  );
};

const styles = StyleSheet.create({
  container: {
    minWidth: 100,
  },
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing.xs,
  },
  label: {
    ...typography.label,
    color: colors.textMuted,
  },
  value: {
    ...typography.h2,
    color: colors.textPrimary,
  },
  valueMono: {
    ...typography.monoLarge,
    color: colors.textPrimary,
  },
});

export default MetricCard;
