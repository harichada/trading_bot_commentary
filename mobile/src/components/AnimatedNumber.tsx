import React, {useEffect} from 'react';
import {TextStyle} from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withTiming,
  withSequence,
  useDerivedValue,
  useAnimatedProps,
  Easing,
} from 'react-native-reanimated';
import {colors, typography} from '../theme';

interface AnimatedNumberProps {
  value: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  style?: TextStyle;
  colorize?: boolean; // green for positive, red for negative
  duration?: number;
}

export function AnimatedNumber({
  value,
  prefix = '',
  suffix = '',
  decimals = 2,
  style,
  colorize = false,
  duration = 600,
}: AnimatedNumberProps) {
  const animatedValue = useSharedValue(value);
  const flashOpacity = useSharedValue(1);
  const prevValue = useSharedValue(value);

  useEffect(() => {
    const changed = prevValue.value !== value;
    prevValue.value = value;

    if (changed) {
      flashOpacity.value = withSequence(
        withTiming(0.4, {duration: 100}),
        withTiming(1, {duration: 300}),
      );
    }
    animatedValue.value = withTiming(value, {
      duration,
      easing: Easing.out(Easing.cubic),
    });
  }, [value]);

  const displayText = useDerivedValue(() => {
    const v = animatedValue.value;
    const sign = v >= 0 && prefix === '' ? '' : '';
    return `${prefix}${sign}${v.toFixed(decimals)}${suffix}`;
  });

  const animatedStyle = useAnimatedStyle(() => {
    const textColor = colorize
      ? animatedValue.value >= 0
        ? colors.green
        : colors.red
      : (style?.color as string) ?? colors.textPrimary;

    return {
      opacity: flashOpacity.value,
      color: textColor,
    };
  });

  const animatedProps = useAnimatedProps(() => {
    return {
      text: displayText.value,
    } as any;
  });

  return (
    <Animated.Text
      style={[typography.mono, style, animatedStyle]}
      animatedProps={animatedProps}
    >
      {`${prefix}${value.toFixed(decimals)}${suffix}`}
    </Animated.Text>
  );
}
