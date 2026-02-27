import React from 'react';
import {View, StyleSheet, ViewStyle} from 'react-native';
import {LinearGradient} from 'expo-linear-gradient';
import {colors, spacing, borderRadius} from '../theme';

interface GradientCardProps {
  children: React.ReactNode;
  accent?: 'green' | 'red' | 'cyan' | 'purple' | 'none';
  style?: ViewStyle;
  padding?: number;
}

const accentColors: Record<string, readonly [string, string]> = {
  green: colors.gradientGreen,
  red: colors.gradientRed,
  cyan: colors.gradientCyan,
  purple: colors.gradientPurple,
  none: colors.gradientCard,
};

const borderColors: Record<string, string> = {
  green: 'rgba(16, 185, 129, 0.2)',
  red: 'rgba(255, 59, 92, 0.2)',
  cyan: 'rgba(0, 212, 255, 0.15)',
  purple: 'rgba(139, 92, 246, 0.15)',
  none: colors.border,
};

export function GradientCard({
  children,
  accent = 'none',
  style,
  padding = spacing.lg,
}: GradientCardProps) {
  const gradientColors = accentColors[accent];
  const borderColor = borderColors[accent];

  return (
    <View style={[styles.container, {borderColor}, style]}>
      <LinearGradient
        colors={[gradientColors[0], gradientColors[1]]}
        start={{x: 0, y: 0}}
        end={{x: 1, y: 1}}
        style={[styles.gradient, {padding}]}
      >
        {children}
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: borderRadius.xl,
    borderWidth: 1,
    overflow: 'hidden',
  },
  gradient: {
    borderRadius: borderRadius.xl - 1,
  },
});
