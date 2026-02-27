import React from 'react';
import {View, Text, ScrollView, StyleSheet} from 'react-native';
import Animated, {FadeInDown} from 'react-native-reanimated';
import {useTradingState} from '../context/TradingStateContext';
import {GradientCard} from '../components/GradientCard';
import {DetailRow} from '../components/DetailRow';
import {EquityChart} from '../components/EquityChart';
import {EmptyState} from '../components/EmptyState';
import {SectionHeader} from '../components/SectionHeader';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TradeRecord} from '../types/trading';

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);

function TradeItem({trade}: {trade: TradeRecord}) {
  const isWin = (trade.pnl ?? 0) >= 0;
  const pnlColor = isWin ? colors.green : colors.red;
  return (
    <View style={styles.tradeRow}>
      <View style={{flex: 1}}>
        <Text style={styles.tradeSymbol}>{trade.symbol}</Text>
        <Text style={styles.tradeDetail}>{trade.side} | {trade.exit_reason}</Text>
      </View>
      <View style={{alignItems: 'flex-end'}}>
        <Text style={[styles.tradePnl, {color: pnlColor}]}>
          {isWin ? '+' : ''}${fmt(trade.pnl)}
        </Text>
        <Text style={[styles.tradeDetail, {color: pnlColor}]}>
          {isWin ? '+' : ''}{fmt(trade.pnl_pct)}%
        </Text>
      </View>
    </View>
  );
}

export function BacktestResultScreen() {
  const {state} = useTradingState();
  const result = state.backtest?.result;

  if (!result) {
    return (
      <View style={[styles.container, {flex: 1}]}>
        <EmptyState icon="bar-chart-outline" title="No Results" subtitle="Run a backtest first to see results" />
      </View>
    );
  }

  const winRate = (result.win_rate ?? 0) * 100;
  const pnl = result.total_pnl ?? 0;
  const pnlColor = pnl >= 0 ? colors.green : colors.red;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* Summary */}
      <Animated.View entering={FadeInDown.duration(300)}>
        <GradientCard accent={pnl >= 0 ? 'green' : 'red'} style={styles.card}>
          <Text style={styles.sectionTitle}>Summary</Text>
          <DetailRow label="Total Trades" value={String(result.total_trades ?? 0)} mono />
          <DetailRow label="Win Rate" value={`${fmt(winRate, 1)}%`} valueColor={winRate >= 50 ? colors.green : colors.red} mono />
          <DetailRow label="Total P&L" value={`$${fmt(pnl, 0)}`} valueColor={pnlColor} mono />
          <DetailRow label="Profit Factor" value={fmt(result.profit_factor)} valueColor={(result.profit_factor ?? 0) >= 1 ? colors.green : colors.red} mono />
          <DetailRow label="Sharpe Ratio" value={fmt(result.sharpe)} valueColor={(result.sharpe ?? 0) >= 1 ? colors.green : colors.yellow} mono />
          <DetailRow label="Max Drawdown" value={`${fmt((result.max_drawdown ?? 0) * 100, 1)}%`} valueColor={colors.red} mono />
        </GradientCard>
      </Animated.View>

      {/* Equity Curve Chart */}
      {result.equity_curve && result.equity_curve.length > 0 && (
        <Animated.View entering={FadeInDown.delay(100).duration(300)}>
          <GradientCard accent="cyan" style={styles.card}>
            <Text style={styles.sectionTitle}>Equity Curve</Text>
            <EquityChart
              equityCurve={result.equity_curve}
              startingEquity={result.equity_curve[0]}
              height={200}
            />
          </GradientCard>
        </Animated.View>
      )}

      {/* Trade Distribution */}
      {result.trades && result.trades.length > 0 && (
        <Animated.View entering={FadeInDown.delay(200).duration(300)}>
          <SectionHeader title="TRADES" count={result.trades.length} defaultExpanded={false}>
            <View style={styles.tradesCard}>
              {result.trades.slice(0, 50).map((t, i) => (
                <TradeItem key={`${t.symbol}-${i}`} trade={t} />
              ))}
              {result.trades.length > 50 && (
                <Text style={styles.moreText}>... and {result.trades.length - 50} more</Text>
              )}
            </View>
          </SectionHeader>
        </Animated.View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg, paddingBottom: spacing.xxxl},
  card: {marginBottom: spacing.md},
  sectionTitle: {...typography.h3, color: colors.textPrimary, marginBottom: spacing.sm},
  tradeRow: {flexDirection: 'row', paddingVertical: spacing.sm, borderBottomWidth: 1, borderBottomColor: colors.border},
  tradeSymbol: {...typography.bodyMedium, color: colors.textPrimary},
  tradeDetail: {...typography.caption, color: colors.textMuted},
  tradePnl: {...typography.monoSmall},
  tradesCard: {
    backgroundColor: colors.bgCard, marginHorizontal: spacing.lg,
    borderRadius: borderRadius.lg, borderWidth: 1, borderColor: colors.border,
    padding: spacing.md, marginBottom: spacing.md,
  },
  moreText: {...typography.caption, color: colors.textMuted, textAlign: 'center', paddingVertical: spacing.sm},
});
