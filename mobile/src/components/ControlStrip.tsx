import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Alert,
  StyleSheet,
} from 'react-native';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TraderStatus} from '../types/trading';

interface ControlStripProps {
  status: TraderStatus;
  onScan: () => void;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onReset: () => void;
}

interface ButtonConfig {
  label: string;
  onPress: () => void;
  color: string;
  bgColor: string;
}

export const ControlStrip: React.FC<ControlStripProps> = ({
  status,
  onScan,
  onStart,
  onStop,
  onPause,
  onResume,
  onReset,
}) => {
  const handleStop = () => {
    Alert.alert(
      'Stop Trading',
      'Are you sure you want to stop the trader? All open positions will remain open.',
      [
        {text: 'Cancel', style: 'cancel'},
        {text: 'Stop', style: 'destructive', onPress: onStop},
      ],
    );
  };

  const handleReset = () => {
    Alert.alert(
      'Reset Trader',
      'This will reset all daily stats and counters. Are you sure?',
      [
        {text: 'Cancel', style: 'cancel'},
        {text: 'Reset', style: 'destructive', onPress: onReset},
      ],
    );
  };

  const getButtons = (): ButtonConfig[] => {
    const buttons: ButtonConfig[] = [];

    // Scan is available in most states
    if (
      status === 'stopped' ||
      status === 'waiting' ||
      status === 'paused'
    ) {
      buttons.push({
        label: 'Scan',
        onPress: onScan,
        color: colors.cyan,
        bgColor: colors.cyanDim,
      });
    }

    // Start when stopped or waiting
    if (status === 'stopped' || status === 'waiting') {
      buttons.push({
        label: 'Start',
        onPress: onStart,
        color: colors.green,
        bgColor: colors.greenDim,
      });
    }

    // Pause when actively trading or scanning
    if (status === 'trading' || status === 'scanning') {
      buttons.push({
        label: 'Pause',
        onPress: onPause,
        color: colors.yellow,
        bgColor: colors.yellowDim,
      });
    }

    // Resume when paused
    if (status === 'paused') {
      buttons.push({
        label: 'Resume',
        onPress: onResume,
        color: colors.green,
        bgColor: colors.greenDim,
      });
    }

    // Stop when trading, scanning, paused, or waiting
    if (
      status === 'trading' ||
      status === 'scanning' ||
      status === 'paused' ||
      status === 'waiting'
    ) {
      buttons.push({
        label: 'Stop',
        onPress: handleStop,
        color: colors.red,
        bgColor: colors.redDim,
      });
    }

    // Reset available when stopped or halted
    if (status === 'stopped' || status === 'halted') {
      buttons.push({
        label: 'Reset',
        onPress: handleReset,
        color: colors.orange,
        bgColor: `${colors.orange}15`,
      });
    }

    return buttons;
  };

  const buttons = getButtons();

  if (buttons.length === 0) {
    return null;
  }

  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.scrollContent}
      style={styles.container}>
      {buttons.map(button => (
        <TouchableOpacity
          key={button.label}
          style={[styles.button, {backgroundColor: button.bgColor}]}
          onPress={button.onPress}
          activeOpacity={0.7}>
          <Text style={[styles.buttonText, {color: button.color}]}>
            {button.label}
          </Text>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flexGrow: 0,
  },
  scrollContent: {
    paddingHorizontal: spacing.lg,
    gap: spacing.sm,
  },
  button: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  buttonText: {
    ...typography.bodyMedium,
  },
});

export default ControlStrip;
