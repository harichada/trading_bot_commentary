# Stage A scorecard r3 — tip `ad05693` (#89+#90 BT emit)

**Owner:** Research | **Date:** 2026-09-17 ET | **Bars:** `/tmp/stage_a_bars_long`  
**Floors (locked):** n≥150 **or** ≥10 sess · PF≥1.30 · WR≥48% · exp≥+0.05R · DD≤6% · max losing day≤2R · max simultaneous≤3 (mean-rev add-on)

**LIVE promote either lane: NO**

---

## 1) mean_reversion LONG — **FAIL / INCOMPLETE**

Source: `/tmp/shadow_long_results/stage_a_summary.json` (promotion_book = risk_off excluded)

| Metric | Value | Floor | Gate |
|--------|------:|------:|------|
| n | 9 | ≥150 | FAIL |
| sessions | 6 | ≥10 | FAIL |
| PF | **0.24** | ≥1.30 | FAIL |
| WR | **33.3%** | ≥48% | FAIL |
| exp_R | **−0.62** | ≥+0.05 | FAIL |
| total_R | −5.58 | — | — |
| maxDD | 7.29R (pct gate true via 0% path — treat R DD as bad) | ≤6% | soft |
| max_losing_day_R | **2.93** | ≤2.0 | FAIL |
| max_simultaneous | 3 | ≤3 | PASS |
| exits | stop×3 · flatten×3 · timeout×3 | — | — |

`all_gates_pass`: **false**. BT unknown-regime 6/6 losers (−7.29R); live seed risk_on 3/3 +1.71R.

**LIVE rec:** **OFF** — do not enable MEAN_REV_LIVE.

---

## 2) day_trade_momentum SHORT — **FAIL NO_GO**

Source: `/tmp/day_trade_short_results/stage_a_summary.json`

| Metric | Value | Floor | Gate |
|--------|------:|------:|------|
| n | 94 | ≥150 | FAIL n (sessions OK) |
| sessions | **18** | ≥10 | PASS |
| PF | **1.35** | ≥1.30 | PASS |
| WR | **39.6%** | ≥48% | FAIL |
| exp_R | **+0.17** | ≥+0.05 | PASS |
| total_R | +15.9 | — | — |
| maxDD | **44.7%** / 7.5R | ≤6% | FAIL |
| max_losing_day_R | **4.0** | ≤2.0 | FAIL |
| max_simultaneous | **10** | ≤3 | FAIL |
| pattern | continuation_down 92 / rejection 2 | — | — |
| exits | stop 49 · target 24 · timeout 11 · flatten 10 | — | — |

`all_gates_pass`: **false**. Sessions+PF+exp green is **not** enough — WR/DD/day-loss/simultaneity fail hard.

**LIVE rec:** **OFF** — keep `DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED=False`.

---

## 3) Other lanes (unchanged)

| Lane | Verdict | LIVE |
|------|---------|------|
| mean_reversion SHORT | FAIL NO_GO (prior: n=88 PF0.78 WR42% exp−0.11) | OFF |
| day_trade_momentum LONG | INCOMPLETE (no BT cut this r3) | ops interim only — not Stage A PASS |
| ORB | INCOMPLETE deferred | OFF |

---

## Research ask
- No LIVE flips.
- Optional: DT SHORT pattern filter (rejection-only) as shadow experiment — still needs Stage A floors before LIVE.
- Grow LONG n only if quality improves; current BT sample is deeply negative PF.
