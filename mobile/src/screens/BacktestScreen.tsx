import React, {useState, useEffect} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  StyleSheet,
} from 'react-native';
import {useNavigation} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {GradientCard} from '../components/GradientCard';
import {ProgressRing} from '../components/ProgressRing';
import {HapticButton} from '../components/HapticButton';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {MoreStackParamList} from '../navigation/stacks/MoreStack';
import Toast from 'react-native-toast-message';

type Nav = NativeStackNavigationProp<MoreStackParamList, 'Backtest'>;

export function BacktestScreen() {
  const navigation = useNavigation<Nav>();
  const {state} = useTradingState();
  const {apiClient} = useAuth();

  const [symbols, setSymbols] = useState('');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [running, setRunning] = useState(false);

  const bt = state.backtest;
  const progress = bt?.progress ?? 0;
  const btRunning = bt?.running ?? false;
  const btMessage = bt?.message ?? '';

  useEffect(() => {
    if (bt?.result && !bt.running && progress >= 100) {
      navigation.navigate('BacktestResult');
    }
  }, [bt?.result, bt?.running, progress, navigation]);

  const handleRun = async () => {
    setRunning(true);
    try {
      await apiClient.runBacktest({
        symbols: symbols || undefined,
        start_date: startDate || undefined,
        end_date: endDate || undefined,
      });
      Toast.show({type: 'info', text1: 'Backtest started'});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Error', text2: err.message});
    } finally {
      setRunning(false);
    }
  };

  const handleCancel = async () => {
    try {
      await apiClient.cancelBacktest();
      Toast.show({type: 'info', text1: 'Backtest cancelled'});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Error', text2: err.message});
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Form */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <GradientCard accent="cyan" style={styles.card}>
          <Text style={styles.sectionTitle}>Backtest Parameters</Text>

          <Text style={styles.label}>Symbols (comma-separated, empty = all)</Text>
          <TextInput
            style={styles.input}
            value={symbols}
            onChangeText={setSymbols}
            placeholder="AAPL, TSLA, NVDA"
            placeholderTextColor={colors.textMuted}
            autoCapitalize="characters"
            editable={!btRunning}
          />

          <Text style={styles.label}>Start Date</Text>
          <TextInput
            style={styles.input}
            value={startDate}
            onChangeText={setStartDate}
            placeholder="YYYY-MM-DD (empty = default)"
            placeholderTextColor={colors.textMuted}
            editable={!btRunning}
          />

          <Text style={styles.label}>End Date</Text>
          <TextInput
            style={styles.input}
            value={endDate}
            onChangeText={setEndDate}
            placeholder="YYYY-MM-DD (empty = today)"
            placeholderTextColor={colors.textMuted}
            editable={!btRunning}
          />
        </GradientCard>
      </Animated.View>

      {/* Progress */}
      {btRunning && (
        <Animated.View entering={FadeInDown.delay(100).duration(300)}>
          <GradientCard accent="purple" style={styles.card}>
            <View style={styles.progressRow}>
              <ProgressRing
                progress={progress}
                size={72}
                strokeWidth={5}
                color={colors.cyan}
              />
              <View style={styles.progressInfo}>
                <Text style={styles.progressTitle}>Running Backtest</Text>
                <Text style={styles.progressMessage} numberOfLines={2}>{btMessage}</Text>
              </View>
            </View>
          </GradientCard>
        </Animated.View>
      )}

      {/* Buttons */}
      <View style={styles.btnRow}>
        {!btRunning ? (
          <HapticButton
            label={running ? 'Starting...' : 'Run Backtest'}
            icon="play"
            onPress={handleRun}
            disabled={running}
            loading={running}
            size="lg"
            style={{flex: 1}}
          />
        ) : (
          <HapticButton
            label="Cancel"
            icon="close"
            variant="danger"
            onPress={handleCancel}
            size="lg"
            style={{flex: 1}}
          />
        )}

        {bt?.result && !btRunning && (
          <HapticButton
            label="View Results"
            icon="bar-chart"
            variant="secondary"
            onPress={() => navigation.navigate('BacktestResult')}
            size="lg"
            style={{flex: 1}}
          />
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg},
  card: {marginBottom: spacing.md},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.md},
  label: {...typography.caption, color: colors.textSecondary, marginBottom: spacing.xs, marginTop: spacing.sm},
  input: {
    backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.lg, padding: spacing.md, color: colors.textPrimary, ...typography.body,
  },
  progressRow: {flexDirection: 'row', alignItems: 'center', gap: spacing.lg},
  progressInfo: {flex: 1},
  progressTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.xs},
  progressMessage: {...typography.caption, color: colors.textMuted},
  btnRow: {flexDirection: 'row', gap: spacing.sm},
});
