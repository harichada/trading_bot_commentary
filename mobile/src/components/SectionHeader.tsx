import React, {useState} from 'react';
import {TouchableOpacity, Text, View, StyleSheet} from 'react-native';
import Animated, {
  useAnimatedStyle,
  withTiming,
  useSharedValue,
} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {colors, typography, spacing} from '../theme';

interface SectionHeaderProps {
  title: string;
  count?: number;
  collapsible?: boolean;
  defaultExpanded?: boolean;
  onToggle?: (expanded: boolean) => void;
  children?: React.ReactNode;
  rightElement?: React.ReactNode;
}

export function SectionHeader({
  title,
  count,
  collapsible = true,
  defaultExpanded = true,
  onToggle,
  children,
  rightElement,
}: SectionHeaderProps) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const rotation = useSharedValue(defaultExpanded ? 1 : 0);

  const chevronStyle = useAnimatedStyle(() => ({
    transform: [{rotate: `${rotation.value * 180}deg`}],
  }));

  const handleToggle = () => {
    if (!collapsible) return;
    const next = !expanded;
    setExpanded(next);
    rotation.value = withTiming(next ? 1 : 0, {duration: 200});
    onToggle?.(next);
  };

  return (
    <View>
      <TouchableOpacity
        style={styles.header}
        onPress={handleToggle}
        activeOpacity={collapsible ? 0.7 : 1}
        disabled={!collapsible}
      >
        <View style={styles.left}>
          <Text style={styles.title}>{title}</Text>
          {count != null && (
            <View style={styles.countBadge}>
              <Text style={styles.countText}>{count}</Text>
            </View>
          )}
        </View>
        <View style={styles.right}>
          {rightElement}
          {collapsible && (
            <Animated.View style={chevronStyle}>
              <Ionicons
                name="chevron-up"
                size={18}
                color={colors.textMuted}
              />
            </Animated.View>
          )}
        </View>
      </TouchableOpacity>
      {(!collapsible || expanded) && children}
    </View>
  );
}

const styles = StyleSheet.create({
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
  },
  left: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  right: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  title: {
    ...typography.label,
    color: colors.textSecondary,
  },
  countBadge: {
    backgroundColor: colors.cyanMuted,
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: 10,
  },
  countText: {
    ...typography.caption,
    color: colors.cyan,
    fontWeight: '600',
    fontSize: 11,
  },
});
