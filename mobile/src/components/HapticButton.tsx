import React, {useCallback} from 'react';
import {
  Text,
  StyleSheet,
  TouchableOpacity,
  ViewStyle,
  TextStyle,
  ActivityIndicator,
} from 'react-native';
import * as Haptics from 'expo-haptics';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withSpring,
} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {colors, typography, spacing, borderRadius} from '../theme';

type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost';

interface HapticButtonProps {
  label: string;
  onPress: () => void;
  variant?: ButtonVariant;
  icon?: keyof typeof Ionicons.glyphMap;
  disabled?: boolean;
  loading?: boolean;
  size?: 'sm' | 'md' | 'lg';
  style?: ViewStyle;
  hapticStyle?: Haptics.ImpactFeedbackStyle;
}

const variantStyles: Record<ButtonVariant, {bg: string; text: string; border: string}> = {
  primary: {bg: colors.cyan, text: colors.bg, border: colors.cyan},
  secondary: {bg: 'transparent', text: colors.cyan, border: colors.cyan},
  danger: {bg: 'transparent', text: colors.red, border: colors.red},
  ghost: {bg: 'transparent', text: colors.textSecondary, border: 'transparent'},
};

const sizeStyles: Record<string, {paddingV: number; paddingH: number; fontSize: number}> = {
  sm: {paddingV: 6, paddingH: 12, fontSize: 12},
  md: {paddingV: 10, paddingH: 16, fontSize: 14},
  lg: {paddingV: 14, paddingH: 24, fontSize: 16},
};

export function HapticButton({
  label,
  onPress,
  variant = 'primary',
  icon,
  disabled = false,
  loading = false,
  size = 'md',
  style,
  hapticStyle = Haptics.ImpactFeedbackStyle.Medium,
}: HapticButtonProps) {
  const scale = useSharedValue(1);
  const v = variantStyles[variant];
  const s = sizeStyles[size];

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{scale: scale.value}],
  }));

  const handlePressIn = useCallback(() => {
    scale.value = withSpring(0.95, {damping: 15, stiffness: 400});
  }, []);

  const handlePressOut = useCallback(() => {
    scale.value = withSpring(1, {damping: 15, stiffness: 400});
  }, []);

  const handlePress = useCallback(() => {
    Haptics.impactAsync(hapticStyle);
    onPress();
  }, [onPress, hapticStyle]);

  return (
    <Animated.View style={animatedStyle}>
      <TouchableOpacity
        style={[
          styles.button,
          {
            backgroundColor: v.bg,
            borderColor: v.border,
            borderWidth: variant === 'ghost' ? 0 : 1,
            paddingVertical: s.paddingV,
            paddingHorizontal: s.paddingH,
            opacity: disabled ? 0.4 : 1,
          },
          style,
        ]}
        onPress={handlePress}
        onPressIn={handlePressIn}
        onPressOut={handlePressOut}
        disabled={disabled || loading}
        activeOpacity={0.8}
      >
        {loading ? (
          <ActivityIndicator size="small" color={v.text} />
        ) : (
          <>
            {icon && (
              <Ionicons
                name={icon}
                size={s.fontSize + 2}
                color={v.text}
                style={styles.icon}
              />
            )}
            <Text style={[styles.label, {color: v.text, fontSize: s.fontSize}]}>
              {label}
            </Text>
          </>
        )}
      </TouchableOpacity>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  button: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: borderRadius.lg,
    gap: spacing.sm,
  },
  icon: {},
  label: {
    fontFamily: typography.bodyMedium.fontFamily,
    fontWeight: '600',
  },
});
