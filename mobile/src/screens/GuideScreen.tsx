import React from 'react';
import {ScrollView, Text, View, StyleSheet} from 'react-native';
import {colors, typography, spacing, borderRadius} from '../theme';

interface GuideSectionProps {
  title: string;
  content: string;
}

function GuideSection({title, content}: GuideSectionProps) {
  return (
    <View style={styles.card}>
      <Text style={styles.sectionTitle}>{title}</Text>
      <Text style={styles.body}>{content}</Text>
    </View>
  );
}

export function GuideScreen() {
  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Gap Fade Strategy Guide</Text>

      <GuideSection
        title="What is Gap Fading?"
        content="Gap fading exploits the statistical tendency for stocks that gap significantly at the open to revert toward the previous close. The bot identifies stocks with large premarket gaps, enters positions to fade (trade against) the gap, and manages risk with automated stops and targets."
      />

      <GuideSection
        title="How Scanning Works"
        content="The scanner runs before market open, looking for stocks with gaps exceeding the configured threshold. It filters by volume ratio, price, shortability, and optionally checks for catalysts (earnings, news). Candidates are scored and ranked."
      />

      <GuideSection
        title="Entry Logic"
        content="For gap-up shorts: the bot waits for the stock to show weakness after the open, then enters a short position. For gap-down longs: it waits for a bounce signal. Position size is calculated using Kelly fraction and risk percentage."
      />

      <GuideSection
        title="Stop Management"
        content="Initial stops are set based on the gap size. Adaptive stops tighten as the trade progresses. The high-water mark tracks the best unrealized P&L, and trailing stops protect profits."
      />

      <GuideSection
        title="Profit Taking"
        content="Partial profits are taken at the half-target (covering a fraction of the position). The remaining position targets full gap fill. Time-based exits close positions before EOD to avoid overnight risk."
      />

      <GuideSection
        title="Circuit Breakers"
        content="The bot halts trading when daily loss limits are hit, consecutive losses exceed the threshold, or drawdown exceeds the maximum. This prevents catastrophic losses on bad days."
      />

      <GuideSection
        title="Rudra LLM Supervisor"
        content="When enabled, the LLM supervisor (Rudra) provides real-time analysis on whether to hold or exit positions. It considers market context, position performance, and risk metrics. You can also chat with Rudra for market insights."
      />

      <GuideSection
        title="Backtesting"
        content="Run historical backtests using OHLCV data stored in the local database. Adjust parameters and compare results. The backtest engine simulates the full trading logic including stops, targets, and circuit breakers."
      />

      <GuideSection
        title="Configuration Tips"
        content="Start with conservative settings: gap_threshold 3-5%, risk_pct 1-2%, max_positions 3-5. Enable adaptive stops and regime filter for better risk management. Use the catalyst detector to avoid earnings gaps."
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  content: {padding: spacing.lg},
  title: {...typography.h1, color: colors.textPrimary, marginBottom: spacing.lg},
  card: {backgroundColor: colors.bgCard, borderRadius: borderRadius.lg, padding: spacing.md, borderWidth: 1, borderColor: colors.border, marginBottom: spacing.md},
  sectionTitle: {...typography.bodyMedium, color: colors.cyan, marginBottom: spacing.sm},
  body: {...typography.body, color: colors.textSecondary, lineHeight: 22},
});
