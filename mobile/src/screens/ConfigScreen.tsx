import React, {useState, useEffect, useCallback, useMemo} from 'react';
import {
  View,
  Text,
  ScrollView,
  TextInput,
  Switch,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import {Ionicons} from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import {useTradingState} from '../context/TradingStateContext';
import {useAuth} from '../context/AuthContext';
import {SectionHeader} from '../components/SectionHeader';
import {HapticButton} from '../components/HapticButton';
import {colors, typography, spacing, borderRadius} from '../theme';
import type {GapFadeConfig} from '../types/trading';
import Toast from 'react-native-toast-message';

type FieldDef = {
  key: keyof GapFadeConfig;
  label: string;
  type: 'number' | 'boolean' | 'string' | 'select';
  options?: string[];
};

type Section = {
  title: string;
  icon: keyof typeof Ionicons.glyphMap;
  fields: FieldDef[];
};

const SECTIONS: Section[] = [
  {title: 'Gap Detection', icon: 'search', fields: [
    {key: 'gap_threshold', label: 'Gap Threshold %', type: 'number'},
    {key: 'max_gap_pct', label: 'Max Gap %', type: 'number'},
    {key: 'vol_ratio_max', label: 'Vol Ratio Max', type: 'number'},
    {key: 'min_avg_volume', label: 'Min Avg Volume', type: 'number'},
    {key: 'min_price', label: 'Min Price $', type: 'number'},
  ]},
  {title: 'Position Sizing', icon: 'resize', fields: [
    {key: 'initial_capital', label: 'Initial Capital $', type: 'number'},
    {key: 'risk_pct', label: 'Risk %', type: 'number'},
    {key: 'kelly_fraction', label: 'Kelly Fraction', type: 'number'},
    {key: 'max_positions', label: 'Max Positions', type: 'number'},
    {key: 'max_notional', label: 'Max Notional $', type: 'number'},
    {key: 'thin_day_threshold', label: 'Thin Day Threshold', type: 'number'},
  ]},
  {title: 'Stops & Targets', icon: 'shield-checkmark', fields: [
    {key: 'stop_pct', label: 'Stop %', type: 'number'},
    {key: 'partial_target_pct', label: 'Partial Target %', type: 'number'},
    {key: 'partial_cover_frac', label: 'Partial Cover Fraction', type: 'number'},
    {key: 'bounce_entry_pct', label: 'Bounce Entry %', type: 'number'},
  ]},
  {title: 'Adaptive Stops', icon: 'analytics', fields: [
    {key: 'adaptive_stops', label: 'Adaptive Stops', type: 'boolean'},
    {key: 'stop_gap_fraction', label: 'Stop Gap Fraction', type: 'number'},
    {key: 'stop_min_pct', label: 'Stop Min %', type: 'number'},
    {key: 'stop_max_pct', label: 'Stop Max %', type: 'number'},
  ]},
  {title: 'Market Regime Filter', icon: 'trending-up', fields: [
    {key: 'regime_filter', label: 'Regime Filter', type: 'boolean'},
    {key: 'regime_spy_gap_limit', label: 'SPY Gap Limit %', type: 'number'},
    {key: 'regime_spy_block_pct', label: 'SPY Block %', type: 'number'},
    {key: 'regime_vix_threshold', label: 'VIX Threshold', type: 'number'},
  ]},
  {title: 'Re-Entry', icon: 'refresh', fields: [
    {key: 'reentry_enabled', label: 'Re-Entry Enabled', type: 'boolean'},
    {key: 'reentry_cooldown_minutes', label: 'Cooldown (min)', type: 'number'},
    {key: 'reentry_max_per_symbol', label: 'Max Per Symbol', type: 'number'},
    {key: 'reentry_stop_pct', label: 'Re-Entry Stop %', type: 'number'},
    {key: 'reentry_trigger_pct', label: 'Re-Entry Trigger %', type: 'number'},
  ]},
  {title: 'Gap-Down Fading (Longs)', icon: 'arrow-down', fields: [
    {key: 'trade_gap_downs', label: 'Trade Gap Downs', type: 'boolean'},
    {key: 'gap_down_threshold', label: 'Gap Down Threshold %', type: 'number'},
    {key: 'gap_down_max_pct', label: 'Gap Down Max %', type: 'number'},
    {key: 'gap_down_vol_ratio_max', label: 'Gap Down Vol Ratio Max', type: 'number'},
  ]},
  {title: 'Entry & Exit Timing', icon: 'time', fields: [
    {key: 'entry_cutoff_hour', label: 'Entry Cutoff Hour', type: 'number'},
    {key: 'entry_cutoff_min', label: 'Entry Cutoff Min', type: 'number'},
    {key: 'min_hold_minutes', label: 'Min Hold (min)', type: 'number'},
    {key: 'min_profit_take_pct', label: 'Min Profit Take %', type: 'number'},
    {key: 'min_gap_fill_pct', label: 'Min Gap Fill %', type: 'number'},
    {key: 'time_exit_hour', label: 'Time Exit Hour', type: 'number'},
    {key: 'time_exit_min', label: 'Time Exit Min', type: 'number'},
    {key: 'eod_exit_hour', label: 'EOD Exit Hour', type: 'number'},
    {key: 'eod_exit_min', label: 'EOD Exit Min', type: 'number'},
  ]},
  {title: 'Circuit Breakers', icon: 'warning', fields: [
    {key: 'daily_loss_limit', label: 'Daily Loss Limit $', type: 'number'},
    {key: 'max_consec_losses', label: 'Max Consec Losses', type: 'number'},
    {key: 'max_drawdown', label: 'Max Drawdown', type: 'number'},
  ]},
  {title: 'Execution', icon: 'flash', fields: [
    {key: 'limit_orders_only', label: 'Limit Orders Only', type: 'boolean'},
    {key: 'limit_offset_pct', label: 'Limit Offset %', type: 'number'},
    {key: 'slippage_pct', label: 'Slippage %', type: 'number'},
    {key: 'borrow_rate_annual', label: 'Borrow Rate (annual)', type: 'number'},
    {key: 'max_pct_adv', label: 'Max % ADV', type: 'number'},
    {key: 'adverse_fill', label: 'Adverse Fill', type: 'boolean'},
    {key: 'adverse_fill_pct', label: 'Adverse Fill %', type: 'number'},
  ]},
  {title: 'Catalyst Detection', icon: 'newspaper', fields: [
    {key: 'catalyst_enabled', label: 'Catalyst Enabled', type: 'boolean'},
    {key: 'catalyst_skip_earnings', label: 'Skip Earnings', type: 'boolean'},
    {key: 'catalyst_earnings_penalty', label: 'Earnings Penalty', type: 'number'},
    {key: 'catalyst_news_penalty', label: 'News Penalty', type: 'number'},
    {key: 'catalyst_noise_bonus', label: 'Noise Bonus', type: 'number'},
  ]},
  {title: 'Scanner', icon: 'globe', fields: [
    {key: 'scan_universe', label: 'Scan Universe', type: 'select', options: ['alpaca', 'study', 'custom']},
    {key: 'custom_symbols', label: 'Custom Symbols', type: 'string'},
    {key: 'backtest_years', label: 'Backtest Years', type: 'number'},
  ]},
  {title: 'Automation & Alerts', icon: 'notifications', fields: [
    {key: 'auto_start', label: 'Auto Start', type: 'boolean'},
    {key: 'alert_telegram_enabled', label: 'Telegram Alerts', type: 'boolean'},
    {key: 'alert_telegram_token', label: 'Telegram Token', type: 'string'},
    {key: 'alert_telegram_chat_id', label: 'Telegram Chat ID', type: 'string'},
  ]},
  {title: 'LLM Supervisor', icon: 'hardware-chip', fields: [
    {key: 'llm_enabled', label: 'LLM Enabled', type: 'boolean'},
    {key: 'llm_url', label: 'LLM URL', type: 'string'},
    {key: 'llm_model', label: 'LLM Model', type: 'string'},
    {key: 'llm_timeout', label: 'LLM Timeout (s)', type: 'number'},
    {key: 'llm_max_failures', label: 'Max Failures', type: 'number'},
    {key: 'llm_circuit_reset', label: 'Circuit Reset (s)', type: 'number'},
    {key: 'llm_max_hold_overrides', label: 'Max Hold Overrides', type: 'number'},
  ]},
];

const PRESETS: Record<string, Partial<GapFadeConfig>> = {
  Conservative: {risk_pct: 0.5, max_positions: 3, stop_pct: 1.0, daily_loss_limit: 500, max_drawdown: 0.05},
  Moderate: {risk_pct: 1.0, max_positions: 5, stop_pct: 1.5, daily_loss_limit: 1000, max_drawdown: 0.08},
  Aggressive: {risk_pct: 2.0, max_positions: 8, stop_pct: 2.0, daily_loss_limit: 2000, max_drawdown: 0.12},
};

export function ConfigScreen() {
  const {state} = useTradingState();
  const {apiClient} = useAuth();
  const [draft, setDraft] = useState<Record<string, any>>({});
  const [saving, setSaving] = useState(false);
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (state.config) setDraft({...state.config});
  }, [state.config]);

  const setValue = useCallback((key: string, value: any) => {
    setDraft(prev => ({...prev, [key]: value}));
  }, []);

  const applyPreset = (name: string) => {
    const preset = PRESETS[name];
    if (preset) {
      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
      setDraft(prev => ({...prev, ...preset}));
      Toast.show({type: 'info', text1: `${name} preset applied`});
    }
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await apiClient.updateStrategyConfig(draft as Partial<GapFadeConfig>);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      Toast.show({type: 'success', text1: 'Configuration saved'});
    } catch (err: any) {
      Toast.show({type: 'error', text1: 'Save failed', text2: err.message});
    } finally {
      setSaving(false);
    }
  };

  const filteredSections = useMemo(() => {
    if (!search.trim()) return SECTIONS;
    const q = search.toLowerCase();
    return SECTIONS.map(s => ({
      ...s,
      fields: s.fields.filter(f => f.label.toLowerCase().includes(q) || f.key.toLowerCase().includes(q)),
    })).filter(s => s.fields.length > 0);
  }, [search]);

  if (!state.config || Object.keys(draft).length === 0) {
    return (
      <View style={[styles.container, {justifyContent: 'center', alignItems: 'center'}]}>
        <ActivityIndicator size="large" color={colors.cyan} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Search */}
      <View style={styles.searchBar}>
        <Ionicons name="search" size={18} color={colors.textMuted} />
        <TextInput
          style={styles.searchInput}
          value={search}
          onChangeText={setSearch}
          placeholder="Search parameters..."
          placeholderTextColor={colors.textMuted}
        />
        {search.length > 0 && (
          <TouchableOpacity onPress={() => setSearch('')}>
            <Ionicons name="close-circle" size={18} color={colors.textMuted} />
          </TouchableOpacity>
        )}
      </View>

      {/* Presets */}
      {!search && (
        <View style={styles.presetRow}>
          {Object.keys(PRESETS).map(name => (
            <HapticButton
              key={name}
              label={name}
              variant="ghost"
              onPress={() => applyPreset(name)}
              size="sm"
              style={styles.presetBtn}
            />
          ))}
        </View>
      )}

      <ScrollView contentContainerStyle={styles.scroll}>
        {filteredSections.map(section => (
          <SectionHeader
            key={section.title}
            title={section.title}
            count={section.fields.length}
            defaultExpanded={!!search}
            rightElement={<Ionicons name={section.icon} size={16} color={colors.textMuted} />}
          >
            <View style={styles.sectionContent}>
              {section.fields.map(field => (
                <View key={field.key} style={styles.fieldRow}>
                  <Text style={styles.fieldLabel}>{field.label}</Text>
                  {field.type === 'boolean' ? (
                    <Switch
                      value={!!draft[field.key]}
                      onValueChange={v => setValue(field.key, v)}
                      trackColor={{false: colors.border, true: `${colors.cyan}60`}}
                      thumbColor={draft[field.key] ? colors.cyan : colors.textMuted}
                    />
                  ) : field.type === 'select' ? (
                    <View style={styles.selectRow}>
                      {field.options?.map(opt => (
                        <TouchableOpacity
                          key={opt}
                          style={[styles.selectBtn, draft[field.key] === opt && styles.selectBtnActive]}
                          onPress={() => setValue(field.key, opt)}>
                          <Text style={[styles.selectText, draft[field.key] === opt && styles.selectTextActive]}>
                            {opt}
                          </Text>
                        </TouchableOpacity>
                      ))}
                    </View>
                  ) : (
                    <TextInput
                      style={styles.fieldInput}
                      value={String(draft[field.key] ?? '')}
                      onChangeText={t => {
                        if (field.type === 'number') {
                          const n = parseFloat(t);
                          setValue(field.key, isNaN(n) ? t : n);
                        } else {
                          setValue(field.key, t);
                        }
                      }}
                      keyboardType={field.type === 'number' ? 'decimal-pad' : 'default'}
                      placeholderTextColor={colors.textMuted}
                    />
                  )}
                </View>
              ))}
            </View>
          </SectionHeader>
        ))}
      </ScrollView>

      {/* Floating save button */}
      <View style={styles.saveContainer}>
        <HapticButton
          label={saving ? 'Saving...' : 'Save Configuration'}
          icon="save"
          onPress={handleSave}
          disabled={saving}
          loading={saving}
          size="lg"
          style={styles.saveBtn}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {flex: 1, backgroundColor: colors.bg},
  searchBar: {
    flexDirection: 'row', alignItems: 'center', gap: spacing.sm,
    marginHorizontal: spacing.lg, marginTop: spacing.sm,
    backgroundColor: colors.bgCard, borderRadius: borderRadius.lg,
    borderWidth: 1, borderColor: colors.border, paddingHorizontal: spacing.md,
  },
  searchInput: {flex: 1, paddingVertical: spacing.sm, color: colors.textPrimary, ...typography.body},
  presetRow: {flexDirection: 'row', justifyContent: 'center', gap: spacing.sm, paddingVertical: spacing.sm, paddingHorizontal: spacing.lg},
  presetBtn: {borderWidth: 1, borderColor: colors.border},
  scroll: {paddingBottom: 100},
  sectionContent: {
    backgroundColor: colors.bgCard, marginHorizontal: spacing.lg,
    borderRadius: borderRadius.lg, borderWidth: 1, borderColor: colors.border,
    marginBottom: spacing.md, overflow: 'hidden',
  },
  fieldRow: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    paddingHorizontal: spacing.md, paddingVertical: spacing.sm,
    borderBottomWidth: 1, borderBottomColor: colors.border,
  },
  fieldLabel: {...typography.caption, color: colors.textSecondary, flex: 1},
  fieldInput: {
    backgroundColor: colors.bg, borderWidth: 1, borderColor: colors.border,
    borderRadius: borderRadius.sm, paddingHorizontal: spacing.sm, paddingVertical: 4,
    color: colors.textPrimary, ...typography.monoSmall, width: 120, textAlign: 'right',
  },
  selectRow: {flexDirection: 'row', gap: 4},
  selectBtn: {paddingHorizontal: spacing.sm, paddingVertical: 4, borderRadius: borderRadius.sm, borderWidth: 1, borderColor: colors.border},
  selectBtnActive: {borderColor: colors.cyan, backgroundColor: `${colors.cyan}20`},
  selectText: {...typography.caption, color: colors.textMuted},
  selectTextActive: {color: colors.cyan},
  saveContainer: {
    position: 'absolute', bottom: spacing.lg, left: spacing.lg, right: spacing.lg,
  },
  saveBtn: {width: '100%'},
});
