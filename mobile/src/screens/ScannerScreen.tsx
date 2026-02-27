import React, {useCallback, useState} from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  RefreshControl,
  ScrollView,
} from 'react-native';
import {useNavigation} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import * as Haptics from 'expo-haptics';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {FilterChips} from '../components/FilterChips';
import {ProgressRing} from '../components/ProgressRing';
import {GradientCard} from '../components/GradientCard';
import {EmptyState} from '../components/EmptyState';
import {HapticButton} from '../components/HapticButton';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {GapCandidate} from '../types/trading';
import type {ScannerStackParamList} from '../navigation/stacks/ScannerStack';
import Toast from 'react-native-toast-message';

type Nav = NativeStackNavigationProp<ScannerStackParamList, 'Scanner'>;

const FILTERS = ['All', 'Gap Up', 'Gap Down', 'High Score', 'ETB Only'];

function CandidateRow({
  candidate,
  onPress,
  index,
}: {
  candidate: GapCandidate;
  onPress: () => void;
  index: number;
}) {
  const dirColor = candidate.direction === 'short' ? colors.red : colors.green;
  const gapColor = candidate.gap_pct > 0 ? colors.green : colors.red;
  const accent = candidate.direction === 'short' ? 'red' as const : 'green' as const;
  const scoreColor = candidate.score >= 7 ? colors.green : candidate.score >= 4 ? colors.yellow : colors.red;

  return (
    <Animated.View entering={FadeInDown.delay(index * 60).duration(250)}>
      <TouchableOpacity onPress={onPress} activeOpacity={0.7}>
        <GradientCard accent={accent} style={rowStyles.card} padding={spacing.md}>
          <View style={rowStyles.topRow}>
            <View style={rowStyles.symbolRow}>
              <Text style={rowStyles.symbol}>{candidate.symbol}</Text>
              <View style={[rowStyles.dirBadge, {backgroundColor: `${dirColor}20`}]}>
                <Text style={[rowStyles.dirText, {color: dirColor}]}>
                  {candidate.direction.toUpperCase()}
                </Text>
              </View>
              {candidate.catalyst !== '' && (
                <View style={rowStyles.catalystBadge}>
                  <Text style={rowStyles.catalystText}>{candidate.catalyst}</Text>
                </View>
              )}
            </View>
            <ProgressRing
              progress={Math.min(candidate.score * 10, 100)}
              size={38}
              strokeWidth={3}
              color={scoreColor}
              showPercent={false}
              label={candidate.score.toFixed(1)}
            />
          </View>

          <View style={rowStyles.detailRow}>
            <View style={rowStyles.detailCol}>
              <Text style={rowStyles.detailLabel}>Gap</Text>
              <Text style={[rowStyles.detailValue, {color: gapColor}]}>
                {candidate.gap_pct > 0 ? '+' : ''}{candidate.gap_pct.toFixed(2)}%
              </Text>
            </View>
            <View style={rowStyles.detailCol}>
              <Text style={rowStyles.detailLabel}>Pre-Mkt</Text>
              <Text style={rowStyles.detailValue}>${candidate.premarket_price.toFixed(2)}</Text>
            </View>
            <View style={rowStyles.detailCol}>
              <Text style={rowStyles.detailLabel}>Vol Ratio</Text>
              <Text style={rowStyles.detailValue}>{candidate.vol_ratio.toFixed(1)}x</Text>
            </View>
            <View style={rowStyles.detailCol}>
              <Text style={rowStyles.detailLabel}>ETB</Text>
              <Text style={[rowStyles.detailValue, {color: candidate.easy_to_borrow ? colors.green : colors.red}]}>
                {candidate.easy_to_borrow ? 'Yes' : 'No'}
              </Text>
            </View>
          </View>
        </GradientCard>
      </TouchableOpacity>
    </Animated.View>
  );
}

const rowStyles = StyleSheet.create({
  card: {marginBottom: spacing.sm},
  topRow: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.sm},
  symbolRow: {flexDirection: 'row', alignItems: 'center', gap: spacing.sm, flex: 1},
  symbol: {...typography.h3, color: colors.textPrimary},
  dirBadge: {paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm},
  dirText: {...typography.label, fontSize: 10},
  catalystBadge: {backgroundColor: 'rgba(245, 158, 11, 0.15)', paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm},
  catalystText: {...typography.label, fontSize: 9, color: colors.yellow},
  detailRow: {flexDirection: 'row', justifyContent: 'space-between'},
  detailCol: {alignItems: 'center'},
  detailLabel: {...typography.caption, color: colors.textMuted, marginBottom: 2},
  detailValue: {...typography.monoSmall, color: colors.textSecondary},
});

export function ScannerScreen() {
  const navigation = useNavigation<Nav>();
  const {state, refreshState} = useTradingState();
  const {apiClient} = useAuth();
  const [refreshing, setRefreshing] = useState(false);
  const [filter, setFilter] = useState('All');

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    try {
      await apiClient.scan();
      await refreshState();
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch {
    } finally {
      setRefreshing(false);
    }
  }, [apiClient, refreshState]);

  const [scanning, setScanning] = useState(false);

  const handleScan = async () => {
    setScanning(true);
    try {
      await apiClient.scan();
      await refreshState();
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Toast.show({type: 'success', text1: 'Scan complete'});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Scan failed', text2: err.message});
    } finally {
      setScanning(false);
    }
  };

  const filtered = state.candidates.filter(c => {
    switch (filter) {
      case 'Gap Up': return c.gap_pct > 0;
      case 'Gap Down': return c.gap_pct < 0;
      case 'High Score': return c.score >= 6;
      case 'ETB Only': return c.easy_to_borrow;
      default: return true;
    }
  });

  const lastScan = state.last_scan_time || 'Never';

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={colors.cyan} colors={[colors.cyan]} />
        }
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.title}>Scanner</Text>
          <View style={styles.headerMeta}>
            <Text style={styles.metaText}>Last scan: {lastScan}</Text>
            <Text style={styles.metaText}>
              {filtered.length} candidate{filtered.length !== 1 ? 's' : ''}
            </Text>
          </View>
        </View>

        {/* Filter Chips */}
        <FilterChips options={FILTERS} selected={filter} onSelect={setFilter} />

        {/* Candidates */}
        {filtered.length === 0 ? (
          <EmptyState
            icon="search-outline"
            title={scanning ? 'Scanning...' : 'No Candidates Found'}
            subtitle={scanning ? 'Looking for gap candidates' : 'Pull down to scan or tap the button below'}
            actionLabel={scanning ? 'Scanning...' : 'Scan Now'}
            onAction={handleScan}
          />
        ) : (
          <View style={styles.list}>
            {filtered.map((candidate, index) => (
              <CandidateRow
                key={candidate.symbol}
                candidate={candidate}
                index={index}
                onPress={() => navigation.navigate('CandidateDetail', {symbol: candidate.symbol})}
              />
            ))}
          </View>
        )}
      </ScrollView>

      {/* Floating Scan Button */}
      {filtered.length > 0 && (
        <View style={styles.fab}>
          <HapticButton label={scanning ? 'Scanning...' : 'Scan Now'} icon="scan" onPress={handleScan} loading={scanning} disabled={scanning} size="md" />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  scrollContent: {paddingTop: spacing.xl + 40, paddingBottom: 80},
  header: {paddingHorizontal: spacing.lg, marginBottom: spacing.sm},
  title: {...typography.h1, color: colors.textPrimary, marginBottom: spacing.xs},
  headerMeta: {flexDirection: 'row', justifyContent: 'space-between'},
  metaText: {...typography.caption, color: colors.textMuted},
  list: {paddingHorizontal: spacing.lg, paddingTop: spacing.sm},
  fab: {position: 'absolute', bottom: spacing.xl, right: spacing.lg},
});
