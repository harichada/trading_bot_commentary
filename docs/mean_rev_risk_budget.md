# Mean-Reversion Quality Gate & Risk Budget

v-meanrev-quality-budget-2026-09-15

## Overview

This document describes the mean-reversion quality gate and separate risk budget system. These modular features prevent mean-rev from consuming the same risk pool as momentum day-trades and reject low-quality signals.

**LIVE stays off. Do NOT flip MEAN_REV_LIVE / ENABLE_MEAN_REV_SHORT / DAY_TRADE live knobs.**

## Stage A Scorecard (Locked)

Same as ORB/momentum — must be met before any LIVE promotion:

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Sample size | n≥150 resolved OR ≥10 sessions with ≥1 resolved | |
| Profit Factor | PF≥1.30 | Fees+slip on hyp fills |
| Win Rate | WR≥48% | Scratches \|R\|<0.05 out of rate, in n |
| Expectancy | exp≥+0.05R | |
| Drawdown | DD≤6% allocated | Max losing day ≤2.0R |

**Prior live short clusters ~PF 0.69 → floors stay hard.**

These Stage A floors apply to SHORT validation. The quality gate RSI/VWAP thresholds (below) are **additive LONG-only** checks, not replacements for Stage A short scorecard.

## Mean-Rev Quality Add-ons

| Constraint | Config | Default | Notes |
|------------|--------|---------|-------|
| Barriers | stop/target/time-stop 60 bars or 15:55 ET flatten | Strategy-level | |
| Primary book | Exclude risk_off regime | `MEAN_REV_EXCLUDE_RISK_OFF=true` | |
| Secondary book | All regimes (shadow) | Shadow ledger | |
| Max simultaneous shorts | ≤3 | `MAX_CONCURRENT_MEAN_REV_SHORTS=3` | |
| Rising-peak filter | Required for shadow promote | `ENABLE_RISING_PEAK_FILTER=true` | **Pre-existing** (v-rising-peak-filter-2026-06-08) |
| Exclude symbols | MU/HQGE/SPCX (+SNAP if hands-off) | `HANDS_OFF_DENYLIST` | |
| Dedupe | Same symbol <15m | `MEAN_REV_DEDUPE_MINUTES=15` | |

> **Note**: The rising-peak filter (`ENABLE_RISING_PEAK_FILTER`) is a **pre-existing feature** from v-rising-peak-filter-2026-06-08, NOT introduced by this PR. It provides symmetric short-side trend-context filtering (close<SMA50 AND MACD<signal required for SHORT). This PR references it as a Stage A constraint; the filter implementation already exists on tip.

### Shadow Ledger Fields

When `MEAN_REV_SHADOW_LEDGER_ENABLED=true` (default), signals emit:

```
setup_type=mean_rev_short
rsi_14=<value>
bb_distance=<pct from BB lower>
atr=<value>
stop_dist=<value>
rr_ratio=<value>
regime=<trending|choppy|risk_off|unknown>
shadow=true
would_be_R=<target R multiple>
```

## Separate Risk Budget

### Config Alias

| Config Name | Research Alias | Semantics |
|-------------|----------------|-----------|
| `MAX_MEAN_REV_RISK_PCT` | `MEAN_REV_RISK_BUDGET_PCT` | Same — max equity % allocated to mean-rev |

Both names are accepted:
- Env: `MEAN_REV_RISK_BUDGET_PCT` (alias) or via yaml
- Yaml: `trading.mean_rev_risk_budget_pct` (alias) or `trading.max_mean_rev_risk_pct`

### Budget Semantics

| Value | Behavior |
|-------|----------|
| **0** (default) | Shadow/paper only — **equity-% check is SKIPPED entirely**. No LIVE risk allocation. Position-count caps still apply. |
| **> 0** | Equity-% check ACTIVE — blocks new entries when (mean-rev notional / equity) >= threshold |

**No contradiction**: When value is 0, the system does NOT perform an equity-% comparison (nothing to compare against). Only position-count caps (MAX_CONCURRENT_MEAN_REV, MAX_CONCURRENT_MEAN_REV_SHORTS) are enforced.

### Config Reference

| Config | Default | Description |
|--------|---------|-------------|
| `ENABLE_MEAN_REV_RISK_BUDGET` | true | Master switch |
| `MAX_CONCURRENT_MEAN_REV` | 3 | Max mean-rev positions (all sides) |
| `MAX_CONCURRENT_MEAN_REV_SHORTS` | 3 | Max simultaneous SHORT positions |
| `MAX_MEAN_REV_RISK_PCT` | **0** | 0 = shadow only; >0 = equity-% cap active |

### Hard Separate Pool

Mean-rev budget is **completely separate** from:
- Day-trade momentum
- ORB (orb_contraction_rvol)
- Future strategies (#2, #3)

Kill-switch flag (`ENABLE_MEAN_REV_RISK_BUDGET`) is **independent** of `DAY_TRADE_LIVE_ENTRIES_ENABLED`.

### LIVE Promotion Path (Hari Decision Required)

1. **Stage A green**: All scorecard thresholds met
2. **Hari approval**: Explicit sign-off required
3. **First LIVE bucket**: ≤1-2% total risk capital only
4. **Mean-rev share**: ≤ half of that unless Hari says otherwise

```yaml
# Example: After Stage A green + Hari approval
trading:
  max_mean_rev_risk_pct: 0.01  # 1% initial allocation
  # OR using Research alias:
  mean_rev_risk_budget_pct: 0.01
```

## Quality Gate (LONG-Only)

**Scope**: This quality gate applies to **LONG entries only** for `mean_reversion` and `oversold_v2` strategies. It is an **additive** engine-level check on top of existing strategy gates.

**NOT Stage A short floors**: The RSI 15-35 / VWAP≤5% thresholds are quality filters for LONG entries. They do NOT replace or modify the Stage A short scorecard (PF≥1.30, WR≥48%, etc.). Stage A short validation uses the locked Research brief metrics.

| Check | Config | Default | Blocks When | Applies To |
|-------|--------|---------|-------------|------------|
| RSI too high | `MEAN_REV_RSI_QUALITY_MAX` | 35.0 | RSI >= threshold | LONG only |
| RSI too low | `MEAN_REV_RSI_QUALITY_MIN` | 15.0 | RSI <= threshold | LONG only |
| VWAP distance | `MEAN_REV_VWAP_DISTANCE_MAX_PCT` | 5.0 | Price > N% below VWAP | LONG only |

Audit reason: `mean_rev_quality_blocked`

## Configuration Reference

### Environment Variables

```bash
# Quality Gate (LONG-only)
ENABLE_MEAN_REV_QUALITY_GATE=1        # Enable/disable (default: on)

# Risk Budget  
ENABLE_MEAN_REV_RISK_BUDGET=1         # Enable/disable (default: on)
MEAN_REV_RISK_BUDGET_PCT=0            # Research alias for MAX_MEAN_REV_RISK_PCT

# Regime Gate
MEAN_REV_EXCLUDE_RISK_OFF=1           # Exclude risk_off regime (default: on)

# Shadow Ledger
MEAN_REV_SHADOW_LEDGER_ENABLED=1      # Emit Stage A fields (default: on)
```

### Config.yaml

```yaml
trading:
  # Quality Gate (LONG-only, additive to strategy gates)
  enable_mean_rev_quality_gate: true
  mean_rev_rsi_quality_max: 35.0      # Max RSI for quality LONG entry
  mean_rev_rsi_quality_min: 15.0      # Min RSI (below = trouble)
  mean_rev_vwap_distance_max_pct: 5.0 # Max % below VWAP
  
  # Risk Budget (separate from day-trade/ORB)
  enable_mean_rev_risk_budget: true
  max_concurrent_mean_rev: 3          # Max mean-rev positions
  max_concurrent_mean_rev_shorts: 3   # Max simultaneous SHORT positions
  # Use EITHER name (same semantics):
  max_mean_rev_risk_pct: 0.0          # 0 = shadow only (LIVE off)
  # mean_rev_risk_budget_pct: 0.0     # Research alias
  
  # Regime Gate
  mean_rev_exclude_risk_off: true     # Primary book excludes risk_off
  
  # Dedupe
  mean_rev_dedupe_minutes: 15         # Same symbol cooldown
  
  # Shadow Ledger
  mean_rev_shadow_ledger_enabled: true # Emit Stage A fields
  
  # Rising-peak filter (PRE-EXISTING, not new in this PR)
  # enable_rising_peak_filter: true   # Already default true since v-rising-peak-filter-2026-06-08
  
  # HANDS_OFF (do not change)
  hands_off_denylist: ['MU', 'HQGE', 'SPCX']
```

## Audit Log Examples

### Quality Block (LONG-only)

```
engine_decision component=mean_rev_quality_gate symbol=AAPL action=blocked \
  reason=mean_rev_quality_blocked strategy=mean_reversion \
  block_reason=rsi_not_oversold rsi=42.5 threshold=35.0
```

### Risk-Off Regime Block

```
engine_decision component=mean_rev_regime_gate symbol=TSLA action=blocked \
  reason=mean_rev_risk_off_blocked strategy=mean_reversion regime=risk_off
```

### Dedupe Block

```
engine_decision component=mean_rev_dedupe_gate symbol=NVDA action=blocked \
  reason=mean_rev_dedupe_blocked strategy=mean_reversion \
  age_min=8.5 dedupe_window_min=15
```

### Budget Exhausted (Position Count)

```
engine_decision component=mean_rev_budget_gate symbol=INTC action=skip \
  reason=mean_rev_budget_exhausted strategy=mean_reversion \
  meanrev_count=3 meanrev_cap=3 meanrev_positions=['NVDA','AMD','TSLA']
```

### Budget Exhausted (Equity %, only when > 0)

```
engine_decision component=mean_rev_budget_gate symbol=QCOM action=skip \
  reason=mean_rev_budget_exhausted strategy=mean_reversion \
  meanrev_notional=15000 equity=50000 meanrev_risk_pct=30.0 max_risk_pct=25.0 \
  reason=risk_pct_exceeded
```

### SHORT Budget Exhausted

```
engine_decision component=mean_rev_budget_gate symbol=IONQ action=skip \
  reason=mean_rev_budget_exhausted strategy=mean_reversion_short \
  meanrev_short_count=3 meanrev_short_cap=3 reason=max_concurrent_shorts_exceeded
```

### HANDS_OFF Block

```
engine_decision component=mean_rev_budget_gate symbol=MU action=skip \
  reason=mean_rev_hands_off_blocked strategy=mean_reversion \
  hands_off_list=['MU', 'HQGE', 'SPCX']
```

### Shadow Ledger Entry

```
engine_decision component=mean_rev_shadow_ledger symbol=PLUG action=shadow \
  reason=stage_a_entry setup_type=mean_rev_buy strategy=mean_reversion \
  entry_pattern=oversold_bounce rsi_14=24.5 bb_distance=-2.3 \
  atr=0.45 stop_dist=1.125 rr_ratio=2.0 regime=choppy \
  shadow=true would_be_R=2.0 entry_price=3.45 stop_loss=2.325 take_profit=5.70
```

## Gate Execution Order

Gates run in the engine signal router in this order:

1. Regime gate (`REGIME_GATE_LIVE_MEANREV`) — trending tape blocks mean-rev
2. Conviction floor (`ENABLE_CONVICTION_FLOOR_MEANREV`) — meta<0.65 blocked
3. **Quality gate** (`ENABLE_MEAN_REV_QUALITY_GATE`) — RSI/VWAP checks (LONG-only)
4. **Risk-off regime gate** (`MEAN_REV_EXCLUDE_RISK_OFF`) — risk_off blocked
5. **Dedupe gate** (`MEAN_REV_DEDUPE_MINUTES`) — same symbol <15m blocked
6. **Shadow ledger** (`MEAN_REV_SHADOW_LEDGER_ENABLED`) — emit Stage A fields
7. ... other gates (health, ML, etc.)
8. **Risk budget** (`ENABLE_MEAN_REV_RISK_BUDGET`) — position/equity caps
   - HANDS_OFF check first
   - Max concurrent shorts check (for SHORT signals)
   - Max concurrent positions check
   - Max equity % check (ONLY if MAX_MEAN_REV_RISK_PCT > 0)

## Safe Off-Path

All features are modular with safe disable:

```bash
# Disable quality gate (mean-rev LONG entries use strategy-level gates only)
ENABLE_MEAN_REV_QUALITY_GATE=0

# Disable risk budget (mean-rev uses global MAX_POSITIONS only)
ENABLE_MEAN_REV_RISK_BUDGET=0

# Disable risk-off exclusion (allow all regimes)
MEAN_REV_EXCLUDE_RISK_OFF=0

# Disable shadow ledger
MEAN_REV_SHADOW_LEDGER_ENABLED=0
```

## Important Notes

1. **LIVE stays stopped**: Does NOT re-enable `MEAN_REV_LIVE_ENTRIES_ENABLED`, `DAY_TRADE_LIVE_ENTRIES_ENABLED`, `ENABLE_MEAN_REV_SHORT`, or any other LIVE knob
2. **HANDS_OFF unchanged**: MU, HQGE, SPCX remain completely untouched
3. **Stage A floors stay hard**: Do NOT soften thresholds — prior live short clusters ~PF 0.69
4. **Quality gate is LONG-only**: RSI 15-35 / VWAP≤5% thresholds do NOT apply to shorts or replace Stage A short scorecard
5. **Rising-peak filter is pre-existing**: `ENABLE_RISING_PEAK_FILTER` exists since v-rising-peak-filter-2026-06-08; this PR references it, does not add it
6. **Budget default 0 = no equity-% check**: When `MAX_MEAN_REV_RISK_PCT=0`, only position-count caps are enforced; equity-% check is skipped
7. **Fail-open**: Missing indicator data (no RSI, no VWAP) does NOT block — only explicit threshold violations
8. **Pending orders counted**: Both open positions AND pending orders count toward the mean-rev budget
9. **Shadow ledger for validation**: Use shadow entries to validate Stage A scorecard before LIVE promotion
