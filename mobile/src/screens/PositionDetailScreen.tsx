import React, {useState} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  StyleSheet,
  Alert,
} from 'react-native';
import {useRoute} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {RouteProp} from '@react-navigation/native';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {GradientCard} from '../components/GradientCard';
import {DetailRow} from '../components/DetailRow';
import {HapticButton} from '../components/HapticButton';
import {EmptyState} from '../components/EmptyState';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {DashboardStackParamList} from '../navigation/stacks/DashboardStack';
import Toast from 'react-native-toast-message';

type Route = RouteProp<DashboardStackParamList, 'PositionDetail'>;

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);

export function PositionDetailScreen() {
  const route = useRoute<Route>();
  const {symbol} = route.params;
  const {state} = useTradingState();
  const {apiClient} = useAuth();
  const position = state.positions[symbol];

  const [newStop, setNewStop] = useState('');
  const [closing, setClosing] = useState(false);
  const [adjusting, setAdjusting] = useState(false);

  if (!position) {
    return (
      <View style={styles.container}>
        <EmptyState icon="briefcase-outline" title={`${symbol} Closed`} subtitle="Position is no longer open" />
      </View>
    );
  }

  const pnlPct = position.high_water_pnl_pct ?? 0;
  const isProfit = pnlPct >= 0;
  const pnlColor = isProfit ? colors.green : colors.red;
  const dirLabel = position.direction === 'short' ? 'SHORT' : 'LONG';
  const dirColor = position.direction === 'short' ? colors.red : colors.green;

  const handleClose = () => {
    Alert.alert('Close Position', `Close ${symbol}?`, [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Close',
        style: 'destructive',
        onPress: async () => {
          setClosing(true);
          try {
            await apiClient.closePosition(symbol);
            Toast.show({type: 'success', text1: `Closing ${symbol}`});
          } catch (err: any) {
            Toast.show({type: 'error', text1: 'Close failed', text2: err.message});
          } finally {
            setClosing(false);
          }
        },
      },
    ]);
  };

  const handleAdjustStop = async () => {
    const price = parseFloat(newStop);
    if (isNaN(price) || price <= 0) {
      Toast.show({type: 'error', text1: 'Enter a valid stop price'});
      return;
    }
    setAdjusting(true);
    try {
      await apiClient.adjustStop(symbol, price);
      setNewStop('');
      Toast.show({type: 'success', text1: `Stop adjusted to $${price.toFixed(2)}`});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Error', text2: err.message});
    } finally {
      setAdjusting(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Header */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <View style={styles.header}>
          <View>
            <Text style={styles.symbol}>{symbol}</Text>
            <View style={[styles.dirBadge, {backgroundColor: `${dirColor}20`}]}>
              <Text style={[styles.dirText, {color: dirColor}]}>{dirLabel}</Text>
            </View>
          </View>
          <View style={{alignItems: 'flex-end'}}>
            <Text style={[styles.pnlValue, {color: pnlColor}]}>
              {isProfit ? '+' : ''}{fmt(pnlPct)}%
            </Text>
            <Text style={styles.pnlLabel}>Unrealized</Text>
          </View>
        </View>
      </Animated.View>

      {/* Details */}
      <Animated.View entering={FadeInDown.delay(100).duration(300)}>
        <GradientCard accent={isProfit ? 'green' : 'red'} style={styles.card}>
          <DetailRow label="Entry Price" value={`$${fmt(position.entry_price)}`} mono />
          <DetailRow label="Fill Price" value={`$${fmt(position.entry_fill_price)}`} mono />
          <DetailRow label="Stop Price" value={`$${fmt(position.stop_price)}`} valueColor={colors.red} mono />
          <DetailRow label="Half Target" value={`$${fmt(position.half_target)}`} mono />
          <DetailRow label="Full Target" value={`$${fmt(position.full_target)}`} valueColor={colors.green} mono />
          <DetailRow label="High Water" value={`$${fmt(position.high_water_price)}`} valueColor={colors.cyan} mono />
          <DetailRow label="Shares" value={String(position.remaining_shares ?? position.shares ?? 0)} mono />
          <DetailRow label="Original Shares" value={String(position.shares ?? 0)} mono />
          <DetailRow label="Partial Filled" value={position.partial_filled ? 'Yes' : 'No'} />
          <DetailRow label="Gap %" value={`${fmt(position.gap_pct, 1)}%`} mono />
          <DetailRow label="Vol Ratio" value={`${fmt(position.vol_ratio, 1)}x`} mono />
          <DetailRow label="Score" value={fmt(position.score, 1)} mono />
          <DetailRow label="Catalyst" value={position.catalyst || 'None'} />
          <DetailRow label="Entry Time" value={position.entry_time || '-'} />
          <DetailRow label="LLM Overrides" value={String(position.llm_hold_overrides ?? 0)} mono />
        </GradientCard>
      </Animated.View>

      {/* Actions */}
      <Animated.View entering={FadeInDown.delay(200).duration(300)}>
        <GradientCard accent="none" style={styles.card}>
          <Text style={styles.sectionTitle}>Actions</Text>
          <HapticButton
            label={closing ? 'Closing...' : 'Close Position'}
            icon="close-circle"
            variant="danger"
            onPress={handleClose}
            disabled={closing}
            loading={closing}
            size="lg"
            style={{marginBottom: spacing.lg}}
          />

          <Text style={styles.fieldLabel}>Adjust Stop Price</Text>
          <View style={styles.adjustRow}>
            <TextInput
              style={styles.input}
              value={newStop}
              onChangeText={setNewStop}
              placeholder={`Current: $${fmt(position.stop_price)}`}
              placeholderTextColor={colors.textMuted}
              keyboardType="decimal-pad"
            />
            <HapticButton
              label={adjusting ? '...' : 'Set'}
              icon="checkmark"
              onPress={handleAdjustStop}
              disabled={adjusting || !newStop}
              size="md"
            />
          </View>
        </GradientCard>
      </Animated.View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingBottom: spacing.xxxl},
  header: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: spacing.lg},
  symbol: {...typography.h1, color: colors.textPrimary},
  dirBadge: {alignSelf: 'flex-start', paddingHorizontal: spacing.sm, paddingVertical: 2, borderRadius: borderRadius.sm, marginTop: spacing.xs},
  dirText: {...typography.label, fontSize: 11},
  pnlValue: {...typography.heroMedium},
  pnlLabel: {...typography.caption, color: colors.textMuted, marginTop: 2},
  card: {marginBottom: spacing.lg},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.md},
  fieldLabel: {...typography.caption, color: colors.textMuted, marginBottom: spacing.xs},
  adjustRow: {flexDirection: 'row', gap: spacing.sm, alignItems: 'center'},
  input: {
    flex: 1, backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.lg, paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
    color: colors.textPrimary, ...typography.mono,
  },
});
