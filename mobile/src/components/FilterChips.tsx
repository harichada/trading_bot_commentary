import React from 'react';
import {ScrollView, TouchableOpacity, Text, StyleSheet} from 'react-native';
import Animated, {FadeIn} from 'react-native-reanimated';
import {colors, typography, spacing, borderRadius} from '../theme';

interface FilterChipsProps {
  options: string[];
  selected: string;
  onSelect: (option: string) => void;
}

export function FilterChips({options, selected, onSelect}: FilterChipsProps) {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.container}
    >
      {options.map((option) => {
        const isActive = option === selected;
        return (
          <Animated.View key={option} entering={FadeIn.duration(200)}>
            <TouchableOpacity
              style={[styles.chip, isActive && styles.chipActive]}
              onPress={() => onSelect(option)}
              activeOpacity={0.7}
            >
              <Text
                style={[styles.chipText, isActive && styles.chipTextActive]}
              >
                {option}
              </Text>
            </TouchableOpacity>
          </Animated.View>
        );
      })}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    gap: spacing.sm,
  },
  chip: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.round,
    backgroundColor: colors.bgCard,
    borderWidth: 1,
    borderColor: colors.border,
  },
  chipActive: {
    backgroundColor: 'rgba(0, 212, 255, 0.12)',
    borderColor: colors.cyan,
  },
  chipText: {
    ...typography.caption,
    color: colors.textSecondary,
    fontFamily: typography.bodyMedium.fontFamily,
  },
  chipTextActive: {
    color: colors.cyan,
  },
});
