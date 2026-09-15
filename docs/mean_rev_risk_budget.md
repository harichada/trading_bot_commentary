# Mean-Reversion Quality Gate & Risk Budget

v-meanrev-quality-budget-2026-09-15

## Overview

This document describes the mean-reversion quality gate and separate risk budget system. These modular features prevent mean-rev from consuming the same risk pool as momentum day-trades and reject low-quality signals.

## Why This Exists

1. **Quality churning**: Mean-rev historically mixed with day-trade risk, producing late-entry churn
2. **Risk concentration**: During volatility spikes, mean-rev can fire on many oversold symbols simultaneously, consuming all MAX_POSITIONS slots
3. **Strategy diversity**: Day-trade momentum needs available slots when movers appear; mean-rev shouldn't block them

## Features

### Quality Gate

Rejects mean-rev entries that don't meet quality thresholds:

| Check | Config | Default | Blocks When |
|-------|--------|---------|-------------|
| RSI too high | `MEAN_REV_RSI_QUALITY_MAX` | 35.0 | RSI >= threshold (not oversold enough) |
| RSI too low | `MEAN_REV_RSI_QUALITY_MIN` | 15.0 | RSI <= threshold (extreme oversold = trouble) |
| VWAP distance | `MEAN_REV_VWAP_DISTANCE_MAX_PCT` | 5.0 | Price > N% below VWAP (chasing extended move) |

Audit reason: `mean_rev_quality_blocked`

### Risk Budget

Separate position cap and equity limit for mean-rev:

| Check | Config | Default | Blocks When |
|-------|--------|---------|-------------|
| Position count | `MAX_CONCURRENT_MEAN_REV` | 3 | Mean-rev positions (incl. pending) >= cap |
| Equity % | `MAX_MEAN_REV_RISK_PCT` | 0.30 | Mean-rev notional / equity >= 30% |

Audit reason: `mean_rev_budget_exhausted`

## Configuration

### Environment Variables

```bash
# Quality Gate
ENABLE_MEAN_REV_QUALITY_GATE=1        # Enable/disable (default: on)

# Risk Budget  
ENABLE_MEAN_REV_RISK_BUDGET=1         # Enable/disable (default: on)
```

### Config.yaml

```yaml
trading:
  # Quality Gate
  enable_mean_rev_quality_gate: true
  mean_rev_rsi_quality_max: 35.0      # Max RSI for quality entry
  mean_rev_rsi_quality_min: 15.0      # Min RSI (below = trouble)
  mean_rev_vwap_distance_max_pct: 5.0 # Max % below VWAP
  
  # Risk Budget
  enable_mean_rev_risk_budget: true
  max_concurrent_mean_rev: 3          # Max mean-rev positions
  max_mean_rev_risk_pct: 0.30         # Max equity % in mean-rev
```

## Ops Tuning Checklist

### When to Tighten Quality Gate

- [ ] Mean-rev producing too many low-quality entries
- [ ] Seeing extended entries (far below VWAP)
- [ ] Extreme oversold entries losing (RSI < 15)

**Actions**:
- Lower `MEAN_REV_RSI_QUALITY_MAX` (e.g., 35 → 30)
- Raise `MEAN_REV_RSI_QUALITY_MIN` (e.g., 15 → 20)
- Lower `MEAN_REV_VWAP_DISTANCE_MAX_PCT` (e.g., 5 → 3)

### When to Loosen Quality Gate

- [ ] Mean-rev blocking too many profitable setups
- [ ] Missing uptrend pullback opportunities

**Actions**:
- Raise `MEAN_REV_RSI_QUALITY_MAX` (e.g., 35 → 40)
- Lower `MEAN_REV_RSI_QUALITY_MIN` (e.g., 15 → 10)
- Raise `MEAN_REV_VWAP_DISTANCE_MAX_PCT` (e.g., 5 → 8)

### When to Tighten Risk Budget

- [ ] Mean-rev consuming too many slots during volatility
- [ ] Day-trade momentum missing opportunities
- [ ] Portfolio too concentrated in mean-rev

**Actions**:
- Lower `MAX_CONCURRENT_MEAN_REV` (e.g., 3 → 2)
- Lower `MAX_MEAN_REV_RISK_PCT` (e.g., 0.30 → 0.20)

### When to Loosen Risk Budget

- [ ] Mean-rev performing well, want more exposure
- [ ] Choppy market where mean-rev is the edge

**Actions**:
- Raise `MAX_CONCURRENT_MEAN_REV` (e.g., 3 → 5)
- Raise `MAX_MEAN_REV_RISK_PCT` (e.g., 0.30 → 0.40)

## Safe Off-Path

Both features are modular with safe disable:

```bash
# Disable quality gate (mean-rev entries use strategy-level gates only)
ENABLE_MEAN_REV_QUALITY_GATE=0

# Disable risk budget (mean-rev uses global MAX_POSITIONS only)
ENABLE_MEAN_REV_RISK_BUDGET=0
```

## Audit Log Examples

### Quality Block

```
engine_decision component=mean_rev_quality_gate symbol=AAPL action=blocked \
  reason=mean_rev_quality_blocked strategy=mean_reversion \
  block_reason=rsi_not_oversold rsi=42.5 threshold=35.0
```

### Budget Exhausted

```
engine_decision component=mean_rev_budget_gate symbol=TSLA action=skip \
  reason=mean_rev_budget_exhausted strategy=mean_reversion \
  meanrev_count=3 meanrev_cap=3 meanrev_positions=['NVDA','AMD','INTC']
```

## Interaction with Other Gates

The quality gate and risk budget run in the engine signal router **after**:
1. Regime gate (`REGIME_GATE_LIVE_MEANREV`)
2. Conviction floor (`ENABLE_CONVICTION_FLOOR_MEANREV`)

And **before**:
1. Global `MAX_POSITIONS` cap
2. Correlation guard
3. Buying power check

The global `MAX_POSITIONS` still applies as a hard ceiling. Mean-rev budget is an additional per-strategy limit.

## Important Notes

1. **LIVE remains stopped**: These gates do not re-enable `MEAN_REV_LIVE_ENTRIES_ENABLED` or any other LIVE knob
2. **HANDS_OFF unchanged**: MU, HQGE, SPCX remain completely untouched
3. **Fail-open**: Missing indicator data (no RSI, no VWAP) does NOT block - only explicit threshold violations
4. **Pending orders counted**: Both open positions AND pending orders count toward the mean-rev budget
