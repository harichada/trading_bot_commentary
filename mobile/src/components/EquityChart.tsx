import React, {useState} from 'react';
import {View, Text, StyleSheet, Dimensions} from 'react-native';
import {LineChart} from 'react-native-wagmi-charts';
import {TouchableOpacity} from 'react-native';
import {colors, typography, spacing, borderRadius} from '../theme';

const SCREEN_WIDTH = Dimensions.get('window').width;

const PERIODS = ['1D', '1W', '1M', '3M', 'ALL'] as const;
type Period = (typeof PERIODS)[number];

interface EquityChartProps {
  equityCurve: number[];
  startingEquity?: number;
  height?: number;
}

export function EquityChart({
  equityCurve,
  startingEquity = 100000,
  height = 180,
}: EquityChartProps) {
  const [period, setPeriod] = useState<Period>('ALL');

  // Build data points from equity curve
  const allData = equityCurve.length > 0
    ? equityCurve.map((val, i) => ({
        timestamp: i,
        value: val,
      }))
    : [{timestamp: 0, value: startingEquity}, {timestamp: 1, value: startingEquity}];

  // Slice data based on period selection
  const sliceData = (data: typeof allData, p: Period) => {
    const len = data.length;
    switch (p) {
      case '1D': return data.slice(Math.max(0, len - 1));
      case '1W': return data.slice(Math.max(0, len - 5));
      case '1M': return data.slice(Math.max(0, len - 22));
      case '3M': return data.slice(Math.max(0, len - 66));
      case 'ALL': return data;
    }
  };

  const chartData = sliceData(allData, period);
  // Ensure minimum 2 data points for the chart
  const displayData = chartData.length < 2
    ? [...chartData, ...(chartData.length === 1 ? [{timestamp: chartData[0].timestamp + 1, value: chartData[0].value}] : [{timestamp: 0, value: startingEquity}, {timestamp: 1, value: startingEquity}])]
    : chartData;

  const firstVal = displayData[0]?.value ?? startingEquity;
  const lastVal = displayData[displayData.length - 1]?.value ?? startingEquity;
  const isPositive = lastVal >= firstVal;
  const lineColor = isPositive ? colors.green : colors.red;

  return (
    <View style={styles.container}>
      <View style={{height}}>
        <LineChart.Provider data={displayData}>
          <LineChart height={height} width={SCREEN_WIDTH - spacing.lg * 2}>
            <LineChart.Path color={lineColor} width={2}>
              <LineChart.Gradient color={lineColor} />
            </LineChart.Path>
            <LineChart.CursorCrosshair color={lineColor}>
              <LineChart.Tooltip
                textStyle={{
                  ...typography.monoSmall,
                  color: colors.textPrimary,
                  backgroundColor: colors.bgCard,
                  padding: 4,
                  borderRadius: 4,
                  overflow: 'hidden',
                }}
              />
            </LineChart.CursorCrosshair>
          </LineChart>
        </LineChart.Provider>
      </View>

      {/* Period Selector */}
      <View style={styles.periodRow}>
        {PERIODS.map((p) => (
          <TouchableOpacity
            key={p}
            style={[styles.periodBtn, period === p && styles.periodBtnActive]}
            onPress={() => setPeriod(p)}
          >
            <Text
              style={[
                styles.periodText,
                period === p && styles.periodTextActive,
              ]}
            >
              {p}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    marginVertical: spacing.sm,
  },
  periodRow: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.xs,
    marginTop: spacing.md,
  },
  periodBtn: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.round,
  },
  periodBtnActive: {
    backgroundColor: 'rgba(0, 212, 255, 0.12)',
  },
  periodText: {
    ...typography.caption,
    color: colors.textMuted,
    fontWeight: '600',
  },
  periodTextActive: {
    color: colors.cyan,
  },
});
