# Stage A Unified Backtest / Walk-Forward Scorecard

**Owner:** Research via CoS  
**Date:** 2026-09-17  
**For:** Hari  
**Type:** Measurement (not a LIVE flip)

---

## Executive Summary

| Lane | Direction | Verdict | LIVE Status | Recommend |
|------|-----------|---------|-------------|-----------|
| 1. mean_reversion LONG | LONG | **INCOMPLETE** | OFF | LIVE off — need resolver |
| 2. day_trade_momentum SHORT | SHORT | **INCOMPLETE** | Shadow (#83) | LIVE off — need shadow sample |
| 3. day_trade_momentum LONG | LONG | **INCOMPLETE** | ON (baseline) | Keep LIVE — pending audit |
| 4. mean_reversion SHORT | SHORT | **FAIL** | OFF | LIVE off — NO_GO confirmed |
| 5. ORB | LONG | **INCOMPLETE** | OFF | LIVE off — not wired |

**Bottom line:**
- **0 lanes PASS** Stage A floors
- **1 lane FAIL** (mean_reversion SHORT — confirmed NO_GO)
- **4 lanes INCOMPLETE** (missing resolver/data/implementation)
- **day_trade_momentum LONG** is currently LIVE as baseline — cannot audit without resolved sample

---

## Locked Stage A Floors (DO NOT SOFTEN)

| Metric | Floor | Definition |
|--------|-------|------------|
| n | ≥ 150 | Closed trades |
| sessions | ≥ 10 | Distinct trading sessions |
| PF | ≥ 1.30 | Profit factor (fees + slippage included) |
| WR | ≥ 48% | Win rate (scratches \|R\|<0.05 excluded from rate, included in n) |
| exp_R | ≥ +0.05 | Expectancy in R-multiples |
| maxDD | ≤ 6% | Max drawdown of allocated capital |
| max_losing_day | ≤ 2.0R | Single session loss cap |

**Hands-off symbols (always excluded):** MU, HQGE, SPCX

**Sample policy:** Primary book = risk_off excluded. Secondary report includes all where data allows.

---

## Priority Lanes (1–3)

### 1. mean_reversion LONG (`oversold_v2`)

**Verdict:** ⚠️ **INCOMPLETE**  
**Current LIVE:** OFF  
**Recommendation:** LIVE off until scorecard available

| Metric | Value |
|--------|-------|
| n_resolved | **unavailable** |
| sessions | **unavailable** |
| PF | **unavailable** |
| WR | **unavailable** |
| exp_R | **unavailable** |
| maxDD | **unavailable** |
| max_losing_day_R | **unavailable** |

**Why INCOMPLETE:**
- Shadow ledger emits exist (`MEAN_REV_SHADOW_LEDGER_ENABLED` → `stage_a_entry` / `setup_type=mean_rev_buy`)
- However: **no LONG barrier resolver** has been run
- `would_be_R` in ledger is target multiple, **not** barrier-resolved R
- Unlike SHORT, no `research/shadow_long_resolver.py` exists

**Engine asks:**
1. Confirm shadow ledger emits on KiddoKingdom for `mean_rev_buy` / `oversold_v2`
2. Implement LONG barrier resolver (or extend SHORT resolver with `--side=LONG`)
3. Run resolve, then apply Stage A floors
4. Until resolved: do not treat any figure as LONG Stage A green/NO_GO

---

### 2. day_trade_momentum SHORT (#83 shadow)

**Verdict:** ⚠️ **INCOMPLETE**  
**Current LIVE:** Shadow only (#83 shadow)  
**Recommendation:** LIVE off until scorecard available

| Metric | Value |
|--------|-------|
| n_resolved | **unavailable** |
| sessions | **unavailable** |
| PF | **unavailable** |
| WR | **unavailable** |
| exp_R | **unavailable** |
| maxDD | **unavailable** |
| max_losing_day_R | **unavailable** |

**Why INCOMPLETE:**
- SHORT signal path exists in `MomentumStrategyWithCommentary` (MACD crossover inversion)
- Not enabled for live execution — shadow only
- No resolved sample in any ledger or Postgres query

**Engine asks:**
1. Implement momentum SHORT shadow ledger writer (analogous to mean_rev shadow)
2. Tag emits with `setup_type=day_trade_continuation`, `direction=SHORT`, `session_id`
3. Implement barrier resolver for momentum shorts
4. Run resolve, then apply Stage A floors

---

### 3. day_trade_momentum LONG (currently LIVE — baseline for keep/kill)

**Verdict:** ⚠️ **INCOMPLETE**  
**Current LIVE:** ON (baseline)  
**Recommendation:** Keep LIVE pending audit — do not scale until Stage A green

| Metric | Value |
|--------|-------|
| n_resolved | **unavailable** |
| sessions | **unavailable** |
| PF | **unavailable** |
| WR | **unavailable** |
| exp_R | **unavailable** |
| maxDD | **unavailable** |
| max_losing_day_R | **unavailable** |

**Why INCOMPLETE:**
- Strategy implemented: `MomentumStrategyWithCommentary` in `trading_bot_commentary_updated.py`
- Currently LIVE — this is the baseline for keep/kill decision
- **No resolved Stage A backtest sample** exists in this harness
- Need: Postgres `minute_bars` with `setup_type=momentum` tagging OR shadow ledger

**Engine asks:**
1. Emit `setup_type=momentum`, `session_id`, `regime` tags on all momentum signals
2. Backfill resolved trades from Schwab fills where `strategy=momentum`
3. Run Stage A scorer against resolved sample
4. If FAIL: escalate keep/kill decision to Hari

---

## Lower Priority Lanes (4–5)

### 4. mean_reversion SHORT — Confirmed NO_GO

**Verdict:** ❌ **FAIL**  
**Current LIVE:** OFF (flags: `ENABLE_MEAN_REV_SHORT=false`, `ENABLE_SHORT_MIRRORS=false`)  
**Recommendation:** LIVE off — shadow soak only until improvement

| Metric | Value | Floor | Status |
|--------|-------|-------|--------|
| n | 88 | ≥ 150 | ❌ |
| sessions | 10 | ≥ 10 | ✅ |
| PF | 0.779 | ≥ 1.30 | ❌ |
| WR | 42.1% | ≥ 48% | ❌ |
| exp_R | −0.11 | ≥ +0.05 | ❌ |
| maxDD | N/A | ≤ 6% | — |
| max_losing_day_R | 13.12R | ≤ 2.0R | ❌ |

**Sample:** 2026-06-11 to 2026-09-08 (historical shadow ledger resolve per 2026-09-11 brief)

**Floor failures (5):**
- n=88 < 150
- PF=0.779 < 1.30
- WR=42.1% < 48%
- exp_R=−0.11 < +0.05
- max_losing_day=13.12R > 2.0R

**Context:**
- Prior live short clusters ~PF 0.69
- Shadow soak only until:
  - Rising-peak filter applied in shadow or live
  - Max simultaneous hyp shorts ≤3 in ≥95% of minutes
  - ≥10 sessions post Config-fix write

**This NO_GO is confirmed. Do not flip live short flags.**

---

### 5. ORB — Not Wired

**Verdict:** ⚠️ **INCOMPLETE**  
**Current LIVE:** OFF  
**Recommendation:** LIVE off — implementation required first

| Metric | Value |
|--------|-------|
| n_resolved | **unavailable** |
| sessions | **unavailable** |
| PF | **unavailable** |
| WR | **unavailable** |
| exp_R | **unavailable** |
| maxDD | **unavailable** |
| max_losing_day_R | **unavailable** |

**Why INCOMPLETE:**
- **ORB strategy class does not exist** in `trading_bot_commentary_updated.py`
- `self.strategies` list in `TradingEngineWithCommentary.__init__` does not include ORB
- 2026-09-10 brief confirms ORB is design-only, not implemented

**Engine asks:**
1. Implement `ORBStrategy` class with:
   - `setup_type=orb` / `orb_15` / `orb_30`
   - OR high/low detection
   - `or_width_atr` calculation
   - Contraction filter
   - `session_id` tagging
2. Wire into `self.strategies` list
3. Run shadow/paper for Stage A sample
4. Stage A cannot proceed until implementation exists

---

## Harness Gaps Summary (Engine Asks)

| Lane | Gap | Ask |
|------|-----|-----|
| mean_reversion LONG | No LONG resolver | Implement `research/shadow_long_resolver.py` or extend SHORT resolver |
| day_trade_momentum SHORT | No shadow ledger | Implement momentum SHORT shadow writer + resolver |
| day_trade_momentum LONG | No resolved sample | Emit `setup_type`, `session_id` tags; backfill from Schwab fills |
| ORB | Not implemented | Implement `ORBStrategy` class and wire into `load_strategies` |

**Infrastructure gaps:**
- No `python -m backtest.cli` with Postgres `minute_bars` integration (expected per task brief)
- No walk-forward or out-of-sample holdout capability — **all metrics in-sample only if/when resolved**
- This harness (`backtest/cli.py`, `backtest/stage_a_scorer.py`) is minimal scaffolding only

---

## Constraints Respected

- [x] Did NOT flip LIVE flags / ENABLE_MEAN_REV_SHORT / DAY_TRADE_LIVE in config defaults
- [x] Did NOT soften locked Stage A floors
- [x] Documented gaps clearly for Engine
- [x] Hands-off excluded: MU, HQGE, SPCX
- [x] mean_reversion SHORT confirmed NO_GO (not fabricated)
- [x] INCOMPLETE lanes honestly marked — no fabricated PF/WR

---

*Scorecard generated by `python3 -m backtest.cli scorecard`*  
*Harness: `/workspace/backtest/cli.py`, `/workspace/backtest/stage_a_scorer.py`*
