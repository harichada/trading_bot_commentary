import React from 'react';
import {View, Text, StyleSheet} from 'react-native';
import ToastMessage, {BaseToast, BaseToastProps} from 'react-native-toast-message';
import {Ionicons} from '@expo/vector-icons';
import {colors, typography, spacing, borderRadius} from '../theme';

const toastConfig = {
  success: (props: BaseToastProps) => (
    <View style={[styles.toast, styles.toastSuccess]}>
      <Ionicons name="checkmark-circle" size={20} color={colors.green} />
      <View style={styles.textContainer}>
        <Text style={styles.title}>{props.text1}</Text>
        {props.text2 && <Text style={styles.subtitle}>{props.text2}</Text>}
      </View>
    </View>
  ),
  error: (props: BaseToastProps) => (
    <View style={[styles.toast, styles.toastError]}>
      <Ionicons name="alert-circle" size={20} color={colors.red} />
      <View style={styles.textContainer}>
        <Text style={styles.title}>{props.text1}</Text>
        {props.text2 && <Text style={styles.subtitle}>{props.text2}</Text>}
      </View>
    </View>
  ),
  info: (props: BaseToastProps) => (
    <View style={[styles.toast, styles.toastInfo]}>
      <Ionicons name="information-circle" size={20} color={colors.cyan} />
      <View style={styles.textContainer}>
        <Text style={styles.title}>{props.text1}</Text>
        {props.text2 && <Text style={styles.subtitle}>{props.text2}</Text>}
      </View>
    </View>
  ),
};

export function ToastProvider({children}: {children: React.ReactNode}) {
  return (
    <>
      {children}
      <ToastMessage config={toastConfig} topOffset={60} />
    </>
  );
}

export {ToastMessage as Toast};

const styles = StyleSheet.create({
  toast: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    marginHorizontal: spacing.lg,
    borderRadius: borderRadius.xl,
    gap: spacing.md,
    borderWidth: 1,
  },
  toastSuccess: {
    backgroundColor: 'rgba(16, 185, 129, 0.12)',
    borderColor: 'rgba(16, 185, 129, 0.25)',
  },
  toastError: {
    backgroundColor: 'rgba(255, 59, 92, 0.12)',
    borderColor: 'rgba(255, 59, 92, 0.25)',
  },
  toastInfo: {
    backgroundColor: 'rgba(0, 212, 255, 0.12)',
    borderColor: 'rgba(0, 212, 255, 0.25)',
  },
  textContainer: {
    flex: 1,
  },
  title: {
    ...typography.bodyMedium,
    color: colors.textPrimary,
  },
  subtitle: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: 2,
  },
});
