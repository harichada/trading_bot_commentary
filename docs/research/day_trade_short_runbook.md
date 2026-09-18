# Day-Trade Momentum SHORT — Research Runbook

**Version:** v-day-trade-short-resolver-2026-09-17  
**Lane:** `day_trade_momentum_short`  
**Direction:** SHORT

---

## Overview

This runbook documents how Research runs Stage A evaluation for the day-trade momentum SHORT strategy. The strategy uses a SHADOW-FIRST deployment model: shadow entries are logged during live operation (without placing actual short orders), then resolved against historical bars to compute Stage A metrics.

## Architecture

```
DayTradeMomentumShortStrategy (strategies/builtin.py)
         │
         ▼ (ENABLE_DAY_TRADE_SHORT_SHADOW=True, DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False)
data/day_trade_short_shadow.ndjson  ← shadow ledger (NDJSON)
         │
         ▼
research/day_trade_short_resolver.py  ← barrier resolver
         │
         ▼
resolved_trades.ndjson + stage_a_summary.json
```

## Prerequisites

1. **Shadow data accumulated** in `data/day_trade_short_shadow.ndjson`
2. **Bars data** available via:
   - Postgres `minute_bars` table (default), OR
   - Local files: `{SYMBOL}.parquet` or `{SYMBOL}.csv` in a bars directory

## Stage A Promotion Floors (DO NOT SOFTEN)

| Metric | Floor |
|--------|-------|
| n (closed trades) | ≥ 150 |
| sessions | ≥ 10 |
| PF (fees+slip) | ≥ 1.30 |
| WR (scratches out) | ≥ 48% |
| exp_R | ≥ +0.05 |
| maxDD | ≤ 6% allocated |
| max_losing_day | ≤ 2.0R |

**Hands-off symbols (excluded):** MU, HQGE, SPCX

---

## Running the Resolver

### Option 1: With Postgres bars (default)

```bash
export POSTGRES_DSN="postgresql://user:pass@host:port/db"

python -m research.day_trade_short_resolver \
    --log data/day_trade_short_shadow.ndjson \
    --out /tmp/day_trade_short_results/
```

### Option 2: With local bars files

Prepare a directory with `{SYMBOL}.parquet` or `{SYMBOL}.csv` files:

```bash
python -m research.day_trade_short_resolver \
    --log data/day_trade_short_shadow.ndjson \
    --bars-dir /path/to/bars/ \
    --out /tmp/day_trade_short_results/
```

### Option 3: Dry run (quick check)

```bash
python -m research.day_trade_short_resolver \
    --log data/day_trade_short_shadow.ndjson \
    --out /tmp/day_trade_short_results/ \
    --dry-run
```

---

## Output Files

| File | Description |
|------|-------------|
| `resolved_trades.ndjson` | One JSON row per resolved trade |
| `resolved_trades.csv` | Same data, tabular format |
| `stage_a_summary.json` | Aggregate metrics with promotion gates |

### stage_a_summary.json structure

```json
{
  "lane": "day_trade_momentum_short",
  "direction": "SHORT",
  "promotion_book": {
    "description": "risk_off excluded",
    "n_trades": 150,
    "n_sessions": 12,
    "win_rate": 0.52,
    "profit_factor": 1.45,
    "expectancy_r": 0.08,
    "by_pattern": { "breakdown": {...}, "continuation_down": {...} },
    "promotion_gates": { "n>=150_or_sessions>=10": true, ... },
    "all_gates_pass": true
  },
  "all_regime": { ... },
  "generated_at": "2026-09-17T..."
}
```

---

## Bars File Format

When using `--bars-dir`, provide files named `{SYMBOL}.parquet` or `{SYMBOL}.csv` with columns:

| Column | Type | Description |
|--------|------|-------------|
| `ts` or `timestamp` | datetime | Bar timestamp (index) |
| `open` | float | Open price |
| `high` | float | High price |
| `low` | float | Low price |
| `close` | float | Close price |
| `volume` | int | Volume |

The resolver filters bars to the window after each shadow entry's timestamp.

---

## Shadow Ledger Format

The shadow ledger (`data/day_trade_short_shadow.ndjson`) contains fields:

| Field | Description |
|-------|-------------|
| `timestamp` | Entry signal time (ISO format) |
| `symbol` | Ticker symbol |
| `signal_type` | Always "SHORT" |
| `strategy` | "day_trade_momentum_short" |
| `entry_pattern` | "breakdown" / "continuation_down" |
| `signal_close` | Price at signal |
| `hypothetical_stop` | Stop loss price (above entry) |
| `hypothetical_target` | Target price (below entry) |
| `stop_dist` | Stop distance in $ |
| `rr_ratio` | Risk/reward ratio |
| `market_context_regime` | "risk_off" / "mixed" / "risk_on" |
| `session_id` | Trading date (YYYY-MM-DD) |

---

## Resolution Logic

1. **Entry price** = `signal_close` + 0.1% slip (conservative adverse for SHORT)
2. **Barriers**: stop (above), target (below)
3. **Time stop**: 60 bars OR flatten at 15:55 ET
4. **R-multiple** = `(entry_price - exit_price) / stop_dist` (positive = profit)

Exit reasons:
- `target` — target hit
- `stop` — stop hit
- `flatten` — EOD flatten at 15:55 ET
- `timeout` — 60 bars without barrier hit
- `no_bars` — no bar data available (excluded from metrics)

---

## Interpreting Results

### Promotion Decision

```
ALL GATES PASS → DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED can be set to True
ANY GATE FAILS → Keep LIVE disabled, continue shadow soak
```

### By Entry Pattern

The resolver breaks down performance by entry pattern:
- **breakdown**: price < 20-bar low with volume
- **continuation_down**: RSI 30-50, price below SMA20, MACD bearish

If one pattern dominates, consider keeping only the profitable pattern.

### By Regime

Performance is broken down by `market_context_regime`:
- **risk_off**: market weak — shorts should perform well
- **mixed**: neutral market
- **risk_on**: excluded from promotion book (shorts blocked in this regime)

---

## Integration with Stage A CLI

The resolver is wired into the Stage A CLI:

```bash
# Generate unified scorecard
python -m backtest.stage_a_cli scorecard --output research/reports/

# Run Stage A for this lane specifically
python -m backtest.stage_a_cli stage-a --lane day_trade_momentum --direction SHORT
```

---

## Checklist Before Enabling LIVE

- [ ] n >= 150 trades OR sessions >= 10
- [ ] PF >= 1.30
- [ ] WR >= 48%
- [ ] exp_R >= +0.05
- [ ] maxDD <= 6%
- [ ] max_losing_day <= 2.0R
- [ ] Manual review of by_pattern breakdown
- [ ] Manual review of by_regime breakdown
- [ ] No anomalous symbols dominating sample

**To enable LIVE:**
```bash
export DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=1
# or in config.yaml:
# trading:
#   day_trade_short_live_entries_enabled: true
```

---

## Historical Sample Generation (BT Sample)

When no live shadow data exists, use the BT sample generator to emit shadow entries from historical bars replay. This creates the INPUT for the resolver.

### Generating Shadow Entries from Historical Bars

**One-liner for Research (file bars):**

```bash
python -m research.day_trade_short_bt_sample \
    --bars-dir /path/to/bars/ \
    --symbols NVDA AAPL TSLA \
    --out data/day_trade_short_shadow.ndjson \
    --allow-risk-on
```

**With Postgres bars:**

```bash
python -m research.day_trade_short_bt_sample \
    --dsn postgresql://... \
    --symbols NVDA AAPL TSLA \
    --days 90 \
    --out data/day_trade_short_shadow.ndjson
```

### BT Sample Options

| Option | Default | Description |
|--------|---------|-------------|
| `--bars-dir` | - | Directory with {SYMBOL}.parquet or .csv files |
| `--dsn` | env/default | Postgres connection string |
| `--symbols` | all in dir | Symbols to process |
| `--days` | 90 | Days of history (Postgres mode) |
| `--start-date` | - | Start date filter (YYYY-MM-DD) |
| `--end-date` | - | End date filter (YYYY-MM-DD) |
| `--out` | data/day_trade_short_shadow.ndjson | Output path |
| `--append` | false | Append to existing file |
| `--min-weak-rs` | 0.5 | Minimum weak RS vs SPY % |
| `--min-volume-ratio` | 1.5 | Minimum volume ratio |
| `--rsi-floor` | 30 | RSI floor (no short below) |
| `--rsi-ceiling` | 80 | RSI ceiling (no short above) |
| `--allow-risk-on` | false | Allow entries in risk_on regime |
| `--spy-bars-file` | - | SPY bars for RS computation |
| `--dry-run` | false | Preview without writing |

### Risk_on Filter

Live strategy HARD BLOCKS entries during risk_on regime (bullish tape). For historical research, you may want to:

1. **Default (risk_on filtered)**: Mimics live behavior exactly.
2. **--allow-risk-on**: Scores setups regardless of regime for full backtest analysis.

Document which mode was used in your research report.

### Full Historical Workflow

```bash
# 1. Generate shadow entries from historical bars
python -m research.day_trade_short_bt_sample \
    --bars-dir /path/to/bars/ \
    --symbols NVDA AAPL TSLA AMZN MSFT \
    --out data/day_trade_short_shadow.ndjson \
    --allow-risk-on

# 2. Resolve entries against bars
python -m research.day_trade_short_resolver \
    --log data/day_trade_short_shadow.ndjson \
    --bars-dir /path/to/bars/ \
    --out /tmp/day_trade_short_results/

# 3. Review Stage A summary
cat /tmp/day_trade_short_results/stage_a_summary.json | jq '.promotion_book'
```

---

## Related Files

| File | Purpose |
|------|---------|
| `strategies/builtin.py` | DayTradeMomentumShortStrategy class |
| `core/config.py` | Config flags (ENABLE_DAY_TRADE_SHORT_SHADOW, etc.) |
| `research/day_trade_short_bt_sample.py` | Historical shadow entry generator |
| `research/day_trade_short_resolver.py` | Barrier resolver for shadow entries |
| `research/shadow_short_resolver.py` | Mean-rev SHORT resolver (similar pattern) |
| `backtest/stage_a_cli.py` | Stage A CLI integration |
| `backtest/stage_a_scorer.py` | Stage A floor definitions |
| `tests/research/test_day_trade_short_bt_sample.py` | BT sample tests |
