import React from 'react';
import {View, Text, StyleSheet, ViewStyle} from 'react-native';
import {colors, typography, spacing} from '../theme';

interface DetailRowProps {
  label: string;
  value: string | number;
  valueColor?: string;
  mono?: boolean;
  style?: ViewStyle;
}

export function DetailRow({
  label,
  value,
  valueColor,
  mono = false,
  style,
}: DetailRowProps) {
  return (
    <View style={[styles.row, style]}>
      <Text style={styles.label}>{label}</Text>
      <Text
        style={[
          mono ? styles.valueMono : styles.value,
          valueColor ? {color: valueColor} : null,
        ]}
      >
        {value}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  label: {
    ...typography.body,
    color: colors.textSecondary,
    flex: 1,
  },
  value: {
    ...typography.bodyMedium,
    color: colors.textPrimary,
    textAlign: 'right',
    flex: 1,
  },
  valueMono: {
    ...typography.mono,
    color: colors.textPrimary,
    textAlign: 'right',
    flex: 1,
  },
});
