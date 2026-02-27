import React, {useEffect} from 'react';
import {View, StyleSheet} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withSequence,
  Easing,
} from 'react-native-reanimated';
import {colors} from '../theme';

interface PulsingDotProps {
  color?: string;
  size?: number;
  active?: boolean;
}

export function PulsingDot({
  color = colors.green,
  size = 8,
  active = true,
}: PulsingDotProps) {
  const pulseScale = useSharedValue(1);
  const pulseOpacity = useSharedValue(0.6);

  useEffect(() => {
    if (active) {
      pulseScale.value = withRepeat(
        withSequence(
          withTiming(2.2, {duration: 1000, easing: Easing.out(Easing.ease)}),
          withTiming(1, {duration: 0}),
        ),
        -1,
      );
      pulseOpacity.value = withRepeat(
        withSequence(
          withTiming(0, {duration: 1000, easing: Easing.out(Easing.ease)}),
          withTiming(0.6, {duration: 0}),
        ),
        -1,
      );
    } else {
      pulseScale.value = 1;
      pulseOpacity.value = 0;
    }
  }, [active]);

  const pulseStyle = useAnimatedStyle(() => ({
    transform: [{scale: pulseScale.value}],
    opacity: pulseOpacity.value,
  }));

  return (
    <View style={[styles.container, {width: size * 3, height: size * 3}]}>
      <Animated.View
        style={[
          styles.pulse,
          pulseStyle,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: color,
          },
        ]}
      />
      <View
        style={[
          styles.dot,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: active ? color : colors.textMuted,
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  pulse: {
    position: 'absolute',
  },
  dot: {},
});
