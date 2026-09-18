# Stage A Unified Backtest / Walk-Forward Scorecard

**Owner:** Research via CoS  
**Date:** 2026-09-17 (r3)  
**For:** Hari  
**Type:** Measurement (not a LIVE flip)  
**As-of tip:** `ad05693` (PR #89+#90 merged — BT emit)  
**See:** `2026-09-17-stage-a-scorecard-r3.md` for full r3 details

---

## Executive Summary

| Lane | Direction | Verdict | LIVE Status | Recommend |
|------|-----------|---------|-------------|-----------|
| 1. mean_reversion LONG | LONG | **FAIL / INCOMPLETE** | OFF | LIVE off — n=9 PF=0.24 WR=33% |
| 2. day_trade_momentum SHORT | SHORT | **FAIL NO_GO** | Shadow | LIVE off — WR/DD/sim fail hard |
| 3. day_trade_momentum LONG | LONG | **INCOMPLETE** | ON (baseline) | Keep LIVE — pending audit |
| 4. mean_reversion SHORT | SHORT | **FAIL NO_GO** | OFF | LIVE off — prior confirmed |
| 5. ORB | LONG | **INCOMPLETE** | OFF | LIVE off — deferred |

**Bottom line:**
- **0 lanes PASS** Stage A floors
- **3 lanes FAIL** (mean_reversion SHORT prior NO_GO, mean_reversion LONG FAIL/INCOMPLETE, day_trade_momentum SHORT NO_GO)
- **2 lanes INCOMPLETE** (day_trade_momentum LONG, ORB)
- **day_trade_momentum LONG** is currently LIVE as baseline — cannot audit without resolved sample
- **No LIVE promotes** — floors locked

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

**Verdict:** ❌ **FAIL / INCOMPLETE**  
**Current LIVE:** OFF  
**Recommendation:** LIVE off — do not enable MEAN_REV_LIVE

| Metric | Value | Floor | Status |
|--------|------:|------:|--------|
| n | 9 | ≥150 | ❌ FAIL |
| sessions | 6 | ≥10 | ❌ FAIL |
| PF | **0.24** | ≥1.30 | ❌ FAIL |
| WR | **33.3%** | ≥48% | ❌ FAIL |
| exp_R | **−0.62** | ≥+0.05 | ❌ FAIL |
| total_R | −5.58 | — | — |
| maxDD | 7.29R | ≤6% | soft |
| max_losing_day_R | **2.93** | ≤2.0 | ❌ FAIL |
| max_simultaneous | 3 | ≤3 | ✅ PASS |

**Source:** `/tmp/shadow_long_results/stage_a_summary.json` (r3 tip `ad05693`)

`all_gates_pass`: **false**. BT unknown-regime 6/6 losers (−7.29R); live seed risk_on 3/3 +1.71R.

**LIVE rec:** **OFF** — do not enable MEAN_REV_LIVE. Grow n only if quality improves; current BT sample is deeply negative PF.

---

### 2. day_trade_momentum SHORT (#83 shadow)

**Verdict:** ❌ **FAIL NO_GO**  
**Current LIVE:** Shadow only  
**Recommendation:** LIVE off — keep `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False`

| Metric | Value | Floor | Status |
|--------|------:|------:|--------|
| n | 94 | ≥150 | ❌ FAIL (sessions OK) |
| sessions | **18** | ≥10 | ✅ PASS |
| PF | **1.35** | ≥1.30 | ✅ PASS |
| WR | **39.6%** | ≥48% | ❌ FAIL |
| exp_R | **+0.17** | ≥+0.05 | ✅ PASS |
| total_R | +15.9 | — | — |
| maxDD | **44.7%** / 7.5R | ≤6% | ❌ FAIL |
| max_losing_day_R | **4.0** | ≤2.0 | ❌ FAIL |
| max_simultaneous | **10** | ≤3 | ❌ FAIL |

**Source:** `/tmp/day_trade_short_results/stage_a_summary.json` (r3 tip `ad05693`)

Pattern: continuation_down 92 / rejection 2  
Exits: stop 49 · target 24 · timeout 11 · flatten 10

`all_gates_pass`: **false**. Sessions+PF+exp green is **not** enough — WR/DD/day-loss/simultaneity fail hard.

**LIVE rec:** **OFF** — keep `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False`. Optional: DT SHORT pattern filter (rejection-only) as shadow experiment — still needs Stage A floors before LIVE.

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

| Lane | Status | Note |
|------|--------|------|
| mean_reversion LONG | ❌ FAIL/INCOMPLETE | n=9 deeply negative PF — grow n only if quality improves |
| day_trade_momentum SHORT | ❌ FAIL NO_GO | WR/DD/sim fail hard — optional rejection-only filter as shadow experiment |
| day_trade_momentum LONG | ⚠️ INCOMPLETE | No BT cut this r3 — ops interim only |
| ORB | ⚠️ INCOMPLETE | Deferred |

**Infrastructure status (PR #89+#90 merged):**
- ✅ `research/shadow_long_resolver.py` — LONG mean-rev barrier resolver
- ✅ `research/shadow_short_resolver.py` — SHORT mean-rev barrier resolver  
- ✅ `research/mean_rev_long_bt_sample.py` — BT emit from bars replay (PR #89)
- ✅ `research/day_trade_short_bt_sample.py` — DT SHORT BT emit (PR #90)
- ✅ `data_providers/file_bars.py` — file-based bars loader + Postgres export CLI
- ⚠️ No walk-forward or out-of-sample holdout capability — **all metrics in-sample only**

---

## Constraints Respected

- [x] Did NOT flip LIVE flags / ENABLE_MEAN_REV_SHORT / DAY_TRADE_LIVE in config defaults
- [x] Did NOT soften locked Stage A floors (n≥150, ≥10 sess, PF≥1.30, WR≥48%, exp≥+0.05R, DD≤6%, max_losing_day≤2R, max_sim≤3)
- [x] Documented gaps clearly for Engine
- [x] Hands-off excluded: MU, HQGE, SPCX (hard-coded in both resolvers)
- [x] mean_reversion SHORT confirmed NO_GO (prior — 5 floor failures)
- [x] mean_reversion LONG FAIL/INCOMPLETE (r3 — n=9 PF=0.24)
- [x] day_trade_momentum SHORT FAIL NO_GO (r3 — WR/DD/sim fail)
- [x] No LIVE promotes — floors locked

---

*Scorecard r3 generated 2026-09-17 after PR #89+#90 merge to tip `ad05693`*  
*Harness: `research/shadow_long_resolver.py`, `research/shadow_short_resolver.py`, `research/mean_rev_long_bt_sample.py`, `research/day_trade_short_bt_sample.py`*
