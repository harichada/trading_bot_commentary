import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {GradientCard} from '../components/GradientCard';
import {DetailRow} from '../components/DetailRow';
import {HapticButton} from '../components/HapticButton';
import {ProgressRing} from '../components/ProgressRing';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {DBStats} from '../types/trading';
import Toast from 'react-native-toast-message';

export function DatabaseScreen() {
  const {apiClient} = useAuth();
  const {state} = useTradingState();
  const [stats, setStats] = useState<DBStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [buildSymbols, setBuildSymbols] = useState('');
  const [buildDate, setBuildDate] = useState('');
  const [building, setBuilding] = useState(false);
  const [updating, setUpdating] = useState(false);

  const dbProgress = state.dbProgress;

  const fetchStats = useCallback(async () => {
    try {
      const data = await apiClient.getDBStats();
      setStats(data);
    } catch {} finally {
      setLoading(false);
    }
  }, [apiClient]);

  useEffect(() => { fetchStats(); }, [fetchStats]);

  const handleBuild = async () => {
    setBuilding(true);
    try {
      await apiClient.buildDB({
        symbols: buildSymbols || undefined,
        start_date: buildDate || undefined,
      });
      Toast.show({type: 'info', text1: 'Database build started'});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Error', text2: err.message});
    } finally {
      setBuilding(false);
    }
  };

  const handleUpdate = async () => {
    setUpdating(true);
    try {
      await apiClient.updateDB();
      Toast.show({type: 'info', text1: 'Database update started'});
      fetchStats();
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Error', text2: err.message});
    } finally {
      setUpdating(false);
    }
  };

  if (loading) {
    return (
      <View style={[styles.container, {justifyContent: 'center', alignItems: 'center'}]}>
        <ActivityIndicator size="large" color={colors.cyan} />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Stats */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <GradientCard accent="cyan" style={styles.card}>
          <Text style={styles.sectionTitle}>Database Stats</Text>
          {stats ? (
            <>
              <DetailRow label="Symbols" value={String(stats.symbol_count ?? 0)} mono />
              <DetailRow label="Total Bars" value={(stats.total_rows ?? 0).toLocaleString()} mono />
              <DetailRow label="Date Range" value={stats.min_date && stats.max_date ? `${stats.min_date} → ${stats.max_date}` : '-'} />
              <DetailRow label="Size" value={`${stats.size_mb ?? 0} MB`} mono />
            </>
          ) : (
            <Text style={styles.emptyText}>Unable to load stats</Text>
          )}
        </GradientCard>
      </Animated.View>

      {/* Progress */}
      {dbProgress?.running && (
        <Animated.View entering={FadeInDown.delay(100).duration(300)}>
          <GradientCard accent="purple" style={styles.card}>
            <View style={styles.progressRow}>
              <ProgressRing progress={dbProgress.progress} size={64} strokeWidth={5} color={colors.cyan} />
              <View style={styles.progressInfo}>
                <Text style={styles.progressTitle}>Building...</Text>
                <Text style={styles.progressMessage} numberOfLines={2}>{dbProgress.message}</Text>
              </View>
            </View>
          </GradientCard>
        </Animated.View>
      )}

      {/* Build */}
      <Animated.View entering={FadeInDown.delay(150).duration(300)}>
        <GradientCard accent="none" style={styles.card}>
          <Text style={styles.sectionTitle}>Build Database</Text>
          <Text style={styles.label}>Symbols (optional)</Text>
          <TextInput
            style={styles.input}
            value={buildSymbols}
            onChangeText={setBuildSymbols}
            placeholder="AAPL, TSLA (empty = all)"
            placeholderTextColor={colors.textMuted}
            autoCapitalize="characters"
          />
          <Text style={styles.label}>Start Date (optional)</Text>
          <TextInput
            style={styles.input}
            value={buildDate}
            onChangeText={setBuildDate}
            placeholder="YYYY-MM-DD"
            placeholderTextColor={colors.textMuted}
          />
          <HapticButton
            label={building ? 'Starting...' : 'Build'}
            icon="hammer"
            onPress={handleBuild}
            disabled={building}
            loading={building}
            size="lg"
            style={{marginTop: spacing.md}}
          />
        </GradientCard>
      </Animated.View>

      {/* Update */}
      <Animated.View entering={FadeInDown.delay(200).duration(300)}>
        <HapticButton
          label={updating ? 'Updating...' : 'Update Database'}
          icon="refresh"
          variant="secondary"
          onPress={handleUpdate}
          disabled={updating}
          loading={updating}
          size="lg"
        />
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingBottom: spacing.xxxl},
  card: {marginBottom: spacing.md},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.sm},
  emptyText: {...typography.body, color: colors.textMuted},
  label: {...typography.caption, color: colors.textSecondary, marginBottom: spacing.xs, marginTop: spacing.sm},
  input: {
    backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.lg, paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
    color: colors.textPrimary, ...typography.body,
  },
  progressRow: {flexDirection: 'row', alignItems: 'center', gap: spacing.lg},
  progressInfo: {flex: 1},
  progressTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.xs},
  progressMessage: {...typography.caption, color: colors.textMuted},
});
