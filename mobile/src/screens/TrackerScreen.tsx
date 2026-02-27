import React, {useState, useCallback, useMemo} from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import {format} from 'date-fns';
import Animated, {FadeInDown} from 'react-native-reanimated';
import {Ionicons} from '@expo/vector-icons';
import {CandlestickChart} from 'react-native-wagmi-charts';
import * as Haptics from 'expo-haptics';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {GradientCard} from '../components/GradientCard';
import {HapticButton} from '../components/HapticButton';
import {EmptyState} from '../components/EmptyState';
import {AnimatedNumber} from '../components/AnimatedNumber';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {TrackerBar} from '../types/trading';
import Toast from 'react-native-toast-message';

const fmt = (v: number | undefined | null, d = 2) => (v ?? 0).toFixed(d);

interface WatchItem {
  symbol: string;
  price: number;
  bid: number;
  ask: number;
}

export function TrackerScreen() {
  const {apiClient} = useAuth();
  const {state} = useTradingState();
  const [symbol, setSymbol] = useState('');
  const [watchlist, setWatchlist] = useState<string[]>([]);
  const [bars, setBars] = useState<TrackerBar[]>([]);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [loadingBars, setLoadingBars] = useState(false);
  const [orderQty, setOrderQty] = useState('');

  const prices = state.trackerPrices ?? {};

  const addSymbol = useCallback(async () => {
    const sym = symbol.trim().toUpperCase();
    if (!sym || watchlist.includes(sym)) {
      setSymbol('');
      return;
    }
    const newList = [...watchlist, sym];
    setWatchlist(newList);
    setSymbol('');
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    try {
      await apiClient.updateWatchlist(newList);
    } catch {}
  }, [symbol, watchlist, apiClient]);

  const removeSymbol = useCallback((sym: string) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    const newList = watchlist.filter(s => s !== sym);
    setWatchlist(newList);
    if (selectedSymbol === sym) {
      setSelectedSymbol('');
      setBars([]);
    }
    apiClient.updateWatchlist(newList).catch(() => {});
  }, [watchlist, selectedSymbol, apiClient]);

  const selectSymbol = useCallback(async (sym: string) => {
    setSelectedSymbol(sym);
    setLoadingBars(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    try {
      const data = await apiClient.getTrackerBars({symbol: sym, limit: 50});
      setBars(Array.isArray(data) ? data : []);
    } catch {
      setBars([]);
    } finally {
      setLoadingBars(false);
    }
  }, [apiClient]);

  const handleOrder = async (side: 'buy' | 'sell') => {
    if (!selectedSymbol) return;
    const qty = parseInt(orderQty, 10);
    if (isNaN(qty) || qty <= 0) {
      Toast.show({type: 'error', text1: 'Invalid quantity'});
      return;
    }
    try {
      await apiClient.placeOrder({symbol: selectedSymbol, qty, side});
      Toast.show({type: 'success', text1: `Order placed`, text2: `${side} ${qty} ${selectedSymbol}`});
      setOrderQty('');
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Order failed', text2: err.message});
    }
  };

  const candlestickData = (bars ?? []).map(b => ({
    timestamp: new Date(b.timestamp).getTime(),
    open: b.open,
    high: b.high,
    low: b.low,
    close: b.close,
  }));

  // Compute price scale (Y-axis) and time labels (X-axis)
  const {priceTicks, timeTicks} = useMemo(() => {
    if (candlestickData.length < 2) return {priceTicks: [], timeTicks: []};
    const highs = candlestickData.map(d => d.high);
    const lows = candlestickData.map(d => d.low);
    const maxP = Math.max(...highs);
    const minP = Math.min(...lows);
    const range = maxP - minP || 1;
    const step = range / 4;
    const pt = Array.from({length: 5}, (_, i) => {
      const price = maxP - i * step;
      const pct = (maxP - price) / range;
      return {price, pct};
    });
    const count = Math.min(4, candlestickData.length);
    const tt = Array.from({length: count}, (_, i) => {
      const idx = Math.round((i / (count - 1)) * (candlestickData.length - 1));
      const ts = candlestickData[idx].timestamp;
      return {label: format(new Date(ts), 'HH:mm'), pct: idx / (candlestickData.length - 1)};
    });
    return {priceTicks: pt, timeTicks: tt};
  }, [candlestickData]);

  const watchItems: WatchItem[] = watchlist.map(sym => ({
    symbol: sym,
    price: prices[sym]?.price ?? 0,
    bid: prices[sym]?.bid ?? 0,
    ask: prices[sym]?.ask ?? 0,
  }));

  const livePrice = prices[selectedSymbol];

  return (
    <View style={styles.container}>
      {/* Search bar */}
      <View style={styles.searchBar}>
        <View style={styles.searchInputWrapper}>
          <Ionicons name="search" size={18} color={colors.textMuted} />
          <TextInput
            style={styles.searchInput}
            value={symbol}
            onChangeText={setSymbol}
            placeholder="Add symbol..."
            placeholderTextColor={colors.textMuted}
            autoCapitalize="characters"
            onSubmitEditing={addSymbol}
            returnKeyType="done"
          />
        </View>
        <HapticButton label="Add" icon="add" onPress={addSymbol} size="sm" />
      </View>

      {watchlist.length === 0 ? (
        <EmptyState
          icon="pulse-outline"
          title="Position Tracker"
          subtitle="Add symbols to track live prices and view candlestick charts"
        />
      ) : (
        <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
          {/* Candlestick Chart */}
          {selectedSymbol && (
            <Animated.View entering={FadeInDown.duration(300)}>
              <GradientCard accent="cyan" style={styles.chartCard}>
                <View style={styles.chartHeader}>
                  <Text style={styles.chartSymbol}>{selectedSymbol}</Text>
                  {livePrice && (
                    <View style={styles.livePriceRow}>
                      <AnimatedNumber
                        value={livePrice.price}
                        prefix="$"
                        style={styles.livePrice}
                        decimals={2}
                      />
                      <Text style={styles.bidAsk}>
                        B: ${fmt(livePrice.bid)} / A: ${fmt(livePrice.ask)}
                      </Text>
                    </View>
                  )}
                </View>

                {loadingBars ? (
                  <ActivityIndicator color={colors.cyan} style={{paddingVertical: spacing.xxl}} />
                ) : candlestickData.length >= 2 ? (
                  <View>
                    {/* Chart + Y-axis */}
                    <View style={styles.chartRow}>
                      <CandlestickChart.Provider data={candlestickData}>
                        <CandlestickChart height={200} width={undefined}>
                          <CandlestickChart.Candles
                            positiveColor={colors.green}
                            negativeColor={colors.red}
                          />
                          <CandlestickChart.Crosshair color={colors.cyan}>
                            <CandlestickChart.Tooltip />
                          </CandlestickChart.Crosshair>
                        </CandlestickChart>
                      </CandlestickChart.Provider>
                      {/* Y-axis price labels */}
                      <View style={styles.yAxis}>
                        {priceTicks.map((tick, i) => (
                          <Text
                            key={i}
                            style={[styles.axisLabel, {top: tick.pct * 200 - 5}]}
                          >
                            {tick.price.toFixed(2)}
                          </Text>
                        ))}
                      </View>
                    </View>
                    {/* X-axis time labels */}
                    <View style={styles.xAxis}>
                      {timeTicks.map((tick, i) => (
                        <Text key={i} style={styles.xAxisLabel}>
                          {tick.label}
                        </Text>
                      ))}
                    </View>
                  </View>
                ) : (
                  <Text style={styles.noBars}>No chart data available</Text>
                )}

                {/* Order Entry */}
                <View style={styles.orderRow}>
                  <TextInput
                    style={styles.qtyInput}
                    value={orderQty}
                    onChangeText={setOrderQty}
                    placeholder="Qty"
                    placeholderTextColor={colors.textMuted}
                    keyboardType="number-pad"
                  />
                  <HapticButton
                    label="Buy"
                    icon="arrow-up"
                    variant="primary"
                    onPress={() => handleOrder('buy')}
                    size="sm"
                    style={{backgroundColor: colors.green, borderColor: colors.green}}
                  />
                  <HapticButton
                    label="Sell"
                    icon="arrow-down"
                    variant="danger"
                    onPress={() => handleOrder('sell')}
                    size="sm"
                  />
                </View>
              </GradientCard>
            </Animated.View>
          )}

          {/* Watchlist */}
          <View style={styles.watchlist}>
            {watchItems.map((item, idx) => (
              <Animated.View key={item.symbol} entering={FadeInDown.delay(idx * 50).duration(200)}>
                <TouchableOpacity
                  style={[styles.watchRow, item.symbol === selectedSymbol && styles.watchRowSelected]}
                  onPress={() => selectSymbol(item.symbol)}
                  activeOpacity={0.7}
                >
                  <View style={styles.watchLeft}>
                    <Text style={styles.watchSymbol}>{item.symbol}</Text>
                    <Text style={styles.watchBidAsk}>
                      {fmt(item.bid)} / {fmt(item.ask)}
                    </Text>
                  </View>
                  <View style={styles.watchRight}>
                    <Text style={styles.watchPrice}>${fmt(item.price)}</Text>
                    <TouchableOpacity onPress={() => removeSymbol(item.symbol)} hitSlop={{top: 10, bottom: 10, left: 10, right: 10}}>
                      <Ionicons name="close-circle" size={18} color={colors.textMuted} />
                    </TouchableOpacity>
                  </View>
                </TouchableOpacity>
              </Animated.View>
            ))}
          </View>
        </ScrollView>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg, paddingTop: spacing.xl + 40},
  searchBar: {flexDirection: 'row', paddingHorizontal: spacing.lg, gap: spacing.sm, marginBottom: spacing.sm, alignItems: 'center'},
  searchInputWrapper: {
    flex: 1, flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    backgroundColor: colors.bgCard, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.lg, paddingHorizontal: spacing.md,
  },
  searchInput: {flex: 1, paddingVertical: spacing.sm, color: colors.textPrimary, ...typography.body},
  scrollContent: {paddingHorizontal: spacing.lg, paddingBottom: spacing.xxxl},
  chartCard: {marginBottom: spacing.md},
  chartRow: {flexDirection: 'row'},
  yAxis: {width: 52, height: 200, position: 'relative'},
  xAxis: {flexDirection: 'row', justifyContent: 'space-between', marginRight: 52, marginTop: 4},
  xAxisLabel: {fontSize: 9, fontFamily: 'Courier', color: colors.textMuted},
  axisLabel: {
    position: 'absolute',
    fontSize: 9,
    fontFamily: 'Courier',
    color: colors.textMuted,
    right: 0,
  },
  chartHeader: {flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: spacing.sm},
  chartSymbol: {...typography.h2, color: colors.textPrimary},
  livePriceRow: {alignItems: 'flex-end'},
  livePrice: {...typography.heroMedium, color: colors.textPrimary},
  bidAsk: {...typography.caption, color: colors.textMuted, marginTop: 2},
  noBars: {...typography.caption, color: colors.textMuted, textAlign: 'center', paddingVertical: spacing.xxl},
  orderRow: {flexDirection: 'row', gap: spacing.sm, marginTop: spacing.md, alignItems: 'center'},
  qtyInput: {
    flex: 1, backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.lg, paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
    color: colors.textPrimary, ...typography.mono, textAlign: 'center',
  },
  watchlist: {gap: spacing.xs},
  watchRow: {
    backgroundColor: colors.bgCard, borderRadius: borderRadius.lg, padding: spacing.md,
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    borderWidth: 1, borderColor: colors.border,
  },
  watchRowSelected: {borderColor: colors.cyan},
  watchLeft: {},
  watchSymbol: {...typography.bodyMedium, color: colors.textPrimary},
  watchBidAsk: {...typography.caption, color: colors.textMuted, marginTop: 2},
  watchRight: {flexDirection: 'row', alignItems: 'center', gap: spacing.md},
  watchPrice: {...typography.monoLarge, color: colors.textPrimary},
});
