import React, {useState, useEffect, useCallback} from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
  RefreshControl,
} from 'react-native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import {useAuth} from '../context/AuthContext';
import {GradientCard} from '../components/GradientCard';
import {DetailRow} from '../components/DetailRow';
import {EmptyState} from '../components/EmptyState';
import {colors, typography, spacing} from '../theme';
import type {AccountInfo} from '../types/trading';

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);

export function AccountScreen() {
  const {apiClient} = useAuth();
  const [account, setAccount] = useState<AccountInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchAccount = useCallback(async () => {
    try {
      const data = await apiClient.getAccount();
      setAccount(data);
    } catch {} finally {
      setLoading(false);
    }
  }, [apiClient]);

  useEffect(() => { fetchAccount(); }, [fetchAccount]);

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await fetchAccount();
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setRefreshing(false);
  }, [fetchAccount]);

  if (loading) {
    return (
      <View style={[styles.container, {justifyContent: 'center', alignItems: 'center'}]}>
        <ActivityIndicator size="large" color={colors.cyan} />
      </View>
    );
  }

  if (!account) {
    return (
      <View style={styles.container}>
        <EmptyState icon="person-outline" title="Account Unavailable" subtitle="Unable to load account information" />
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.cyan} colors={[colors.cyan]} />}
    >
      {/* Equity Hero */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <View style={styles.heroRow}>
          <Text style={styles.heroLabel}>Portfolio Equity</Text>
          <Text style={styles.heroValue}>${fmt(account.equity, 0)}</Text>
        </View>
      </Animated.View>

      {/* Account Details */}
      <Animated.View entering={FadeInDown.delay(100).duration(300)}>
        <GradientCard accent="cyan" style={styles.card}>
          <Text style={styles.sectionTitle}>Alpaca Account</Text>
          <DetailRow label="Equity" value={`$${fmt(account.equity, 0)}`} valueColor={colors.cyan} mono />
          <DetailRow label="Buying Power" value={`$${fmt(account.buying_power, 0)}`} mono />
          <DetailRow label="Cash" value={`$${fmt(account.cash, 0)}`} mono />
          <DetailRow label="Portfolio Value" value={`$${fmt(account.portfolio_value, 0)}`} mono />
          <DetailRow
            label="PDT Status"
            value={account.pattern_day_trader ? 'Yes' : 'No'}
            valueColor={account.pattern_day_trader ? colors.red : colors.green}
          />
        </GradientCard>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingBottom: spacing.xxxl},
  heroRow: {marginBottom: spacing.lg},
  heroLabel: {...typography.label, color: colors.textMuted, marginBottom: spacing.xs},
  heroValue: {...typography.heroLarge, color: colors.textPrimary},
  card: {marginBottom: spacing.md},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.sm},
});
