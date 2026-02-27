import React, {useEffect} from 'react';
import {View, Text, StyleSheet} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TraderStatus} from '../types/trading';

interface StatusBadgeProps {
  status: TraderStatus;
}

const statusColorMap: Record<TraderStatus, string> = {
  trading: colors.green,
  paused: colors.yellow,
  stopped: colors.textMuted,
  scanning: colors.cyan,
  halted: colors.red,
  waiting: colors.yellow,
  standdown: colors.red,
};

const statusLabelMap: Record<TraderStatus, string> = {
  trading: 'Trading',
  paused: 'Paused',
  stopped: 'Stopped',
  scanning: 'Scanning',
  halted: 'Halted',
  waiting: 'Waiting',
  standdown: 'Stand Down',
};

const activeStatuses: TraderStatus[] = ['trading', 'scanning'];

export const StatusBadge: React.FC<StatusBadgeProps> = ({status}) => {
  const color = statusColorMap[status];
  const label = statusLabelMap[status];
  const isActive = activeStatuses.includes(status);
  const dotScale = useSharedValue(1);

  useEffect(() => {
    if (isActive) {
      dotScale.value = withRepeat(
        withSequence(
          withTiming(1.4, {duration: 600}),
          withTiming(1, {duration: 600}),
        ),
        -1,
      );
    } else {
      dotScale.value = 1;
    }
  }, [isActive]);

  const dotStyle = useAnimatedStyle(() => ({
    transform: [{scale: dotScale.value}],
  }));

  return (
    <View style={[styles.container, {backgroundColor: `${color}15`}]}>
      <Animated.View style={[styles.dot, {backgroundColor: color}, dotStyle]} />
      <Text style={[styles.text, {color}]}>{label}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.round,
    alignSelf: 'flex-start',
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: spacing.xs,
  },
  text: {
    ...typography.label,
    textTransform: 'uppercase',
  },
});

export default StatusBadge;
