import React from 'react';
import {View, Text, StyleSheet, Dimensions} from 'react-native';
import {Gesture, GestureDetector} from 'react-native-gesture-handler';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
  runOnJS,
  interpolate,
  Extrapolation,
} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {colors, spacing} from '../theme';

const SWIPE_THRESHOLD = 80;

interface SwipeableRowProps {
  children: React.ReactNode;
  onSwipeRight?: () => void;
  rightLabel?: string;
  rightIcon?: keyof typeof Ionicons.glyphMap;
  rightColor?: string;
  enabled?: boolean;
}

export function SwipeableRow({
  children,
  onSwipeRight,
  rightLabel = 'Close',
  rightIcon = 'close-circle',
  rightColor = colors.red,
  enabled = true,
}: SwipeableRowProps) {
  const translateX = useSharedValue(0);

  const gesture = Gesture.Pan()
    .enabled(enabled)
    .onUpdate((e) => {
      translateX.value = Math.min(0, e.translationX);
    })
    .onEnd((e) => {
      if (e.translationX < -SWIPE_THRESHOLD && onSwipeRight) {
        runOnJS(onSwipeRight)();
      }
      translateX.value = withSpring(0, {damping: 20, stiffness: 200});
    });

  const rowStyle = useAnimatedStyle(() => ({
    transform: [{translateX: translateX.value}],
  }));

  const actionStyle = useAnimatedStyle(() => {
    const width = interpolate(
      translateX.value,
      [-120, 0],
      [120, 0],
      Extrapolation.CLAMP,
    );
    const opacity = interpolate(
      translateX.value,
      [-SWIPE_THRESHOLD, -20],
      [1, 0],
      Extrapolation.CLAMP,
    );
    return {width, opacity};
  });

  return (
    <View style={styles.container}>
      <Animated.View style={[styles.action, {backgroundColor: rightColor}, actionStyle]}>
        <Ionicons name={rightIcon} size={20} color="#fff" />
        <Text style={styles.actionText}>{rightLabel}</Text>
      </Animated.View>
      <GestureDetector gesture={gesture}>
        <Animated.View style={rowStyle}>{children}</Animated.View>
      </GestureDetector>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'relative',
    overflow: 'hidden',
  },
  action: {
    position: 'absolute',
    right: 0,
    top: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    gap: 4,
  },
  actionText: {
    color: '#fff',
    fontSize: 10,
    fontWeight: '600',
  },
});
