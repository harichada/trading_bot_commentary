import React, {useCallback} from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  Alert,
  RefreshControl,
} from 'react-native';
import {useNavigation} from '@react-navigation/native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import type {NativeStackNavigationProp} from '@react-navigation/native-stack';
import {FlashList} from '@shopify/flash-list';
import * as Haptics from 'expo-haptics';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {HeroSection} from '../components/HeroSection';
import {EquityChart} from '../components/EquityChart';
import {MetricCard} from '../components/MetricCard';
import {HapticButton} from '../components/HapticButton';
import {SwipeableRow} from '../components/SwipeableRow';
import {GradientCard} from '../components/GradientCard';
import {SectionHeader} from '../components/SectionHeader';
import {EmptyState} from '../components/EmptyState';
import {ConfirmDialog} from '../components/ConfirmDialog';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {GapPosition} from '../types/trading';
import type {DashboardStackParamList} from '../navigation/stacks/DashboardStack';
import Toast from 'react-native-toast-message';

type Nav = NativeStackNavigationProp<DashboardStackParamList, 'Dashboard'>;

const fmt = (v: number | undefined | null, decimals = 2): string =>
  (v ?? 0).toFixed(decimals);

// ─── Position Card (redesigned) ─────────────────────────────────

function PositionCard({
  position,
  onPress,
  onClose,
  index,
}: {
  position: GapPosition;
  onPress: () => void;
  onClose: () => void;
  index: number;
}) {
  const currentPnlPct = position.high_water_pnl_pct ?? 0;
  const isProfit = currentPnlPct >= 0;
  const pnlColor = isProfit ? colors.green : colors.red;
  const dirLabel = position.direction === 'short' ? 'SHORT' : 'LONG';
  const dirColor = position.direction === 'short' ? colors.red : colors.green;
  const accent = isProfit ? 'green' as const : 'red' as const;

  return (
    <Animated.View entering={FadeInDown.delay(index * 80).duration(300)}>
      <SwipeableRow onSwipeRight={onClose} rightLabel="Close">
        <GradientCard accent={accent} style={posStyles.card} padding={spacing.md}>
          <View style={posStyles.header}>
            <View style={posStyles.symbolRow}>
              <Text style={posStyles.symbol}>{position.symbol}</Text>
              <View style={[posStyles.dirBadge, {backgroundColor: `${dirColor}20`}]}>
                <Text style={[posStyles.dirText, {color: dirColor}]}>{dirLabel}</Text>
              </View>
            </View>
            <Text style={[posStyles.pnl, {color: pnlColor}]}>
              {isProfit ? '+' : ''}{fmt(currentPnlPct)}%
            </Text>
          </View>

          <View style={posStyles.details} onTouchEnd={onPress}>
            <View style={posStyles.detailCol}>
              <Text style={posStyles.detailLabel}>Entry</Text>
              <Text style={posStyles.detailValue}>${fmt(position.entry_price)}</Text>
            </View>
            <View style={posStyles.detailCol}>
              <Text style={posStyles.detailLabel}>Stop</Text>
              <Text style={posStyles.detailValue}>${fmt(position.stop_price)}</Text>
            </View>
            <View style={posStyles.detailCol}>
              <Text style={posStyles.detailLabel}>Shares</Text>
              <Text style={posStyles.detailValue}>{position.remaining_shares ?? 0}</Text>
            </View>
            <View style={posStyles.detailCol}>
              <Text style={posStyles.detailLabel}>Gap</Text>
              <Text style={posStyles.detailValue}>{fmt(position.gap_pct, 1)}%</Text>
            </View>
          </View>
        </GradientCard>
      </SwipeableRow>
    </Animated.View>
  );
}

const posStyles = StyleSheet.create({
  card: {
    marginBottom: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  symbolRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  symbol: {
    ...typography.h3,
    color: colors.textPrimary,
  },
  dirBadge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  dirText: {
    ...typography.label,
    fontSize: 10,
  },
  pnl: {
    ...typography.monoLarge,
  },
  details: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailCol: {
    alignItems: 'center',
  },
  detailLabel: {
    ...typography.caption,
    color: colors.textMuted,
    marginBottom: 2,
  },
  detailValue: {
    ...typography.monoSmall,
    color: colors.textSecondary,
  },
});

// ─── DashboardScreen ─────────────────────────────────────────────

export function DashboardScreen() {
  const navigation = useNavigation<Nav>();
  const {state, refreshState} = useTradingState();
  const {apiClient} = useAuth();
  const [refreshing, setRefreshing] = React.useState(false);
  const [closeSymbol, setCloseSymbol] = React.useState<string | null>(null);

  const positions = Object.values(state.positions);
  const m = state.metrics;

  const dailyPnl = state.daily_stats?.pnl ?? 0;
  const equity = state.equity || state.config?.initial_capital || 100000;
  const dailyPnlPct = equity > 0 ? (dailyPnl / (equity - dailyPnl)) * 100 : 0;

  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refreshState();
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    setRefreshing(false);
  }, [refreshState]);

  const wrap = (label: string, fn: () => Promise<unknown>) => async () => {
    try {
      await fn();
      Toast.show({type: 'success', text1: `${label} successful`});
    } catch (err: any) {
      Toast.show({type: 'error', text1: `${label} failed`, text2: err.message});
    }
  };

  const handleClosePosition = async () => {
    if (!closeSymbol) return;
    try {
      await apiClient.closePosition(closeSymbol);
      Toast.show({type: 'success', text1: `Closing ${closeSymbol}`});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Close failed', text2: err.message});
    }
    setCloseSymbol(null);
  };

  const isTrading = state.status === 'trading';
  const isPaused = state.status === 'paused';

  return (
    <View style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.cyan}
            colors={[colors.cyan]}
          />
        }
      >
        {/* Hero: Equity + P&L + Status */}
        <HeroSection
          equity={equity}
          dailyPnl={dailyPnl}
          dailyPnlPct={dailyPnlPct}
          status={state.status}
          connected={state.connected}
        />

        {/* Equity Sparkline Chart */}
        <Animated.View entering={FadeInDown.delay(100).duration(400)}>
          <EquityChart
            equityCurve={state.backtest?.result?.equity_curve ?? [equity]}
            startingEquity={state.config?.initial_capital ?? 100000}
          />
        </Animated.View>

        {/* Metric Cards */}
        <Animated.View entering={FadeInDown.delay(200).duration(400)}>
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            contentContainerStyle={styles.metricsContent}
          >
            <MetricCard
              label="Win Rate"
              value={`${fmt((m.win_rate ?? 0) * 100, 1)}%`}
              color={(m.win_rate ?? 0) >= 0.5 ? colors.green : colors.red}
              trend={(m.win_rate ?? 0) >= 0.5 ? 'up' : 'down'}
              accent={(m.win_rate ?? 0) >= 0.5 ? 'green' : 'red'}
              mono
            />
            <MetricCard
              label="Total P&L"
              value={`$${fmt(m.total_pnl, 0)}`}
              color={(m.total_pnl ?? 0) >= 0 ? colors.green : colors.red}
              trend={(m.total_pnl ?? 0) >= 0 ? 'up' : 'down'}
              accent={(m.total_pnl ?? 0) >= 0 ? 'green' : 'red'}
              mono
            />
            <MetricCard
              label="Profit Factor"
              value={fmt(m.profit_factor)}
              color={(m.profit_factor ?? 0) >= 1 ? colors.green : colors.red}
              trend={(m.profit_factor ?? 0) >= 1 ? 'up' : 'down'}
              mono
            />
            <MetricCard
              label="Sharpe"
              value={fmt(m.sharpe)}
              color={(m.sharpe ?? 0) >= 1 ? colors.green : colors.yellow}
              trend={(m.sharpe ?? 0) >= 1 ? 'up' : 'neutral'}
              mono
            />
            <MetricCard
              label="Trades"
              value={String(m.total_trades ?? 0)}
              mono
            />
            <MetricCard
              label="Drawdown"
              value={`${fmt((m.max_drawdown ?? 0) * 100, 1)}%`}
              color={colors.red}
              trend="down"
              mono
            />
          </ScrollView>
        </Animated.View>

        {/* Control Strip */}
        <Animated.View entering={FadeInDown.delay(300).duration(400)}>
          <View style={styles.controlStrip}>
            {!isTrading && !isPaused && (
              <HapticButton
                label="Start"
                icon="play"
                variant="primary"
                onPress={wrap('Start', () => apiClient.start())}
                size="sm"
              />
            )}
            {isTrading && (
              <HapticButton
                label="Pause"
                icon="pause"
                variant="secondary"
                onPress={wrap('Pause', () => apiClient.pause())}
                size="sm"
                style={{borderColor: colors.yellow}}
              />
            )}
            {isPaused && (
              <HapticButton
                label="Resume"
                icon="play"
                variant="primary"
                onPress={wrap('Resume', () => apiClient.resume())}
                size="sm"
              />
            )}
            {(isTrading || isPaused) && (
              <HapticButton
                label="Stop"
                icon="stop"
                variant="danger"
                onPress={() => {
                  Alert.alert('Stop Trading', 'Stop the trader? Positions will remain open.', [
                    {text: 'Cancel', style: 'cancel'},
                    {text: 'Stop', style: 'destructive', onPress: wrap('Stop', () => apiClient.stop())},
                  ]);
                }}
                size="sm"
              />
            )}
            <HapticButton
              label="Scan"
              icon="scan"
              variant="secondary"
              onPress={wrap('Scan', () => apiClient.scan())}
              size="sm"
            />
            {isTrading && (
              <HapticButton
                label="Enter"
                icon="log-in"
                variant="secondary"
                onPress={wrap('Enter', () => apiClient.enter())}
                size="sm"
                style={{borderColor: colors.purple}}
              />
            )}
          </View>
        </Animated.View>

        {/* Open Positions */}
        <SectionHeader
          title="OPEN POSITIONS"
          count={positions.length}
          collapsible={false}
        />

        {positions.length === 0 ? (
          <EmptyState
            icon="briefcase-outline"
            title="No Open Positions"
            subtitle="Start trading or scan for candidates to enter positions"
          />
        ) : (
          <View style={styles.positionsList}>
            {positions.map((pos, index) => (
              <PositionCard
                key={pos.symbol}
                position={pos}
                index={index}
                onPress={() => navigation.navigate('PositionDetail', {symbol: pos.symbol})}
                onClose={() => setCloseSymbol(pos.symbol)}
              />
            ))}
          </View>
        )}

        {/* Today's Trades Summary */}
        {state.today_trades.length > 0 && (
          <Animated.View entering={FadeInDown.delay(400).duration(400)}>
            <SectionHeader
              title="TODAY'S TRADES"
              count={state.today_trades.length}
              collapsible
              defaultExpanded={false}
            >
              <View style={styles.todayTrades}>
                {state.today_trades.slice(-5).map((trade, i) => {
                  const isWin = trade.pnl >= 0;
                  return (
                    <View key={`${trade.symbol}-${trade.exit_time}-${i}`} style={styles.tradeRow}>
                      <View style={styles.tradeLeft}>
                        <Text style={styles.tradeSymbol}>{trade.symbol}</Text>
                        <Text style={[styles.tradeSide, {color: trade.side === 'short' ? colors.red : colors.green}]}>
                          {trade.side?.toUpperCase()}
                        </Text>
                      </View>
                      <Text style={[styles.tradePnl, {color: isWin ? colors.green : colors.red}]}>
                        {isWin ? '+' : ''}${fmt(trade.pnl)}
                      </Text>
                    </View>
                  );
                })}
              </View>
            </SectionHeader>
          </Animated.View>
        )}
      </ScrollView>

      {/* Close Position Confirmation */}
      <ConfirmDialog
        visible={closeSymbol !== null}
        title="Close Position"
        message={`Close ${closeSymbol}? This will place a market order to close the position.`}
        confirmText="Close"
        destructive
        onConfirm={handleClosePosition}
        onCancel={() => setCloseSymbol(null)}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.bg,
  },
  scrollContent: {
    padding: spacing.lg,
    paddingTop: spacing.xl + 40,
    paddingBottom: spacing.xxxl,
  },
  metricsContent: {
    gap: spacing.sm,
    paddingVertical: spacing.sm,
  },
  controlStrip: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    paddingVertical: spacing.md,
  },
  positionsList: {
    marginTop: spacing.sm,
  },
  todayTrades: {
    paddingHorizontal: spacing.lg,
    gap: spacing.xs,
  },
  tradeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  tradeLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  tradeSymbol: {
    ...typography.bodyMedium,
    color: colors.textPrimary,
  },
  tradeSide: {
    ...typography.label,
    fontSize: 10,
  },
  tradePnl: {
    ...typography.mono,
  },
});
