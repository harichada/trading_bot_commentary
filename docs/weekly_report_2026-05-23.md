# Weekly Trading Bot Report — generated 2026-05-23 17:11

**Stop patching. Measure honestly. Decide based on data.**

## Section 1 — Headline

- Logs parsed: 6 files
- Strategy signals (any action): **141300**
- Accepted entries (signal_router): **32**
- Closed trades (matched to Schwab fills): **23**
- **Realized P&L (closed long trades): $+1,268.21**
- Win rate: **15/23 = 65.2%**

## Section 2 — Per-strategy P&L

| Strategy | N | WR | PF | Net P&L | Avg Win | Avg Loss | Avg Hold (min) |
|---|---:|---:|---:|---:|---:|---:|---:|
| breakout | 2 | 50.0% | 1.14 | $+38.92 | $+316.12 | $-277.20 | 15.1 |
| mean_reversion | 8 | 62.5% | 2.09 | $+524.53 | $+201.08 | $-160.29 | 24.8 |
| news | 13 | 69.2% | 2.17 | $+704.76 | $+145.39 | $-150.93 | 116.0 |

## Section 3 — Per-symbol-class P&L

| Class | N | WR | PF | Net P&L |
|---|---:|---:|---:|---:|
| leveraged_etf | 1 | 0.0% | 0.0 | $-228.45 |
| mid_cap | 2 | 100.0% | inf | $+374.40 |
| small_or_other | 20 | 65.0% | 1.99 | $+1,122.26 |

## Section 4 — Per-time-slot P&L

| Slot | N | WR | PF | Net P&L | Avg Hold |
|---|---:|---:|---:|---:|---:|
| first_5min | 1 | 0.0% | 0.0 | $-247.72 | 3.0m |
| morning_active | 15 | 60.0% | 1.94 | $+881.83 | 27.0m |
| late_morning | 1 | 100.0% | inf | $+38.99 | 107.0m |
| midday | 5 | 80.0% | 2.69 | $+291.52 | 40.4m |
| close_hour | 1 | 100.0% | inf | $+303.60 | 1019.8m |

## Section 5 — Per-classifier-reading P&L

| Classifier bucket | N | WR | PF | Net P&L |
|---|---:|---:|---:|---:|
| strong_bearish_<0.25 | 5 | 60.0% | 3.64 | $+616.08 |
| bearish_0.25-0.35 | 5 | 60.0% | 1.03 | $+9.14 |
| neutral_0.35-0.40 | 13 | 69.2% | 1.83 | $+642.99 |

## Section 6 — Short-side counterfactual

For every classifier shadow signal with `rule_allowed=['short']` AND `long_score < 0.30`,
simulated a 2%/4% R:R short trade using Schwab 1-min bars (same broker as live trading).
**This is the alpha we left on the table by being long-only.**

- Simulated short trades: **20**
- Hypothetical realized: **$+142.76**
- Win rate: **40.0%**
- Profit factor: **1.1**
- Avg hold: 135.9 min

### Per-strategy-of-original-signal
| Source signal | N | WR | Net P&L |
|---|---:|---:|---:|
| _classifier_short_signal | 20 | 40.0% | $+142.76 |

## Section 7 — Closed trade list

| Sym | Strategy | Entry | Exit | Side | PnL | Hold (min) | Exit | cls_long |
|---|---|---:|---:|---|---:|---:|---|---:|
| RKLB | mean_reversion | $124.29 | $126.88 | BUY | $+300.44 | 1.3 | target | 0.231 |
| MRNA | mean_reversion | $46.03 | $46.02 | BUY | $-4.71 | 30.5 | manual | 0.231 |
| FIG | mean_reversion | $22.54 | $22.16 | BUY | $-247.72 | 3.0 | stop | 0.294 |
| LUNR | mean_reversion | $31.76 | $31.92 | BUY | $+64.32 | 0.6 | manual | 0.345 |
| ROIV | news | $32.44 | $32.56 | BUY | $+38.99 | 107.0 | manual | 0.375 |
| ALAB | news | $276.57 | $281.08 | BUY | $+189.21 | 16.6 | manual | 0.305 |
| LUNR | mean_reversion | $31.77 | $32.44 | BUY | $+187.60 | 6.0 | manual | 0.231 |
| ELF | mean_reversion | $50.53 | $51.37 | BUY | $+91.85 | 88.1 | manual | 0.366 |
| IONQ | news | $56.80 | $58.92 | BUY | $+284.08 | 14.8 | target | 0.377 |
| NBIS | news | $212.53 | $219.44 | BUY | $+103.65 | 7.4 | target | 0.369 |
| STLA | news | $7.07 | $7.20 | BUY | $+114.08 | 57.6 | manual | 0.250 |
| GFS | news | $79.08 | $77.43 | BUY | $-128.31 | 50.0 | stop | 0.366 |
| APLD | breakout | $47.27 | $46.25 | BUY | $-277.20 | 18.8 | stop | 0.380 |
| QBTS | news | $24.48 | $24.84 | BUY | $+153.74 | 33.4 | manual | 0.368 |
| IBM | news | $244.18 | $245.90 | BUY | $+73.96 | 29.8 | manual | 0.362 |
| ASTS | news | $94.62 | $97.26 | BUY | $+303.60 | 1019.8 | target | 0.363 |
| FUTU | breakout | $87.00 | $92.65 | BUY | $+316.12 | 11.4 | manual | 0.379 |
| SEDG | mean_reversion | $59.27 | $62.07 | BUY | $+361.20 | 12.6 | manual | 0.231 |
| SOXS | mean_reversion | $7.87 | $7.68 | BUY | $-228.45 | 56.2 | stop | 0.231 |
| IMAX | news | $39.30 | $38.45 | BUY | $-192.10 | 46.8 | stop | 0.391 |
| WDAY | news | $131.44 | $129.31 | BUY | $-110.75 | 3.1 | stop | 0.309 |
| IONQ | news | $64.86 | $63.85 | BUY | $-172.55 | 57.4 | stop | 0.381 |
| EL | news | $87.00 | $87.66 | BUY | $+47.16 | 64.7 | manual | 0.372 |

## Section 8 — Decisions (locked 2026-05-23, corrected report)

> **Note**: The original 2026-05-22 report had two trades tagged "?" due to a
> 12-second log skew between `strategy_decision signal_buy` and
> `signal_router accepted`, plus a `.env` parser bug that prevented Schwab
> from authenticating during the rerun. Both fixed; this report uses
> correctly-tagged data. **Some PF readings shifted enough to revise
> decisions** (notably: chop-zone veto was attempted, reverted on review).

### Strategy roster for Tuesday 2026-05-26 open

| Strategy | Status | Size | Rationale |
|---|---|---|---|
| **news** | **LIVE** | full | PF 2.17, N=13, WR 69% — still the primary alpha (gap to mean_rev narrowed vs original 5/22 read) |
| **mean_reversion** | **WATCH (half-size)** | 50% | PF 2.09, N=8 — slightly better than original read but still under-sampled. Soak until N≥20 |
| **breakout** | **DISABLED** | 0 | PF 1.14 on N=2 cannot be distinguished from luck. Confirmed by APLD -$277 vs FUTU +$316 variance |
| **untagged ("?")** | **N/A** | 0 | Was a measurement bug, not a strategy. Fixed in `research/weekly_report.py` (30s join + reason fallback + alias normalization) |

### Time gate

- **NO ENTRIES BEFORE 9:35 ET.** first_5min slot: -$247 on FIG (3-min stop-out).
- morning_active (9:35–10:30) is the prime window: PF 1.94, N=15, +$881.
- Implemented in `core/engine.py:4676-4707` — hard block, no meta_proba bypass.

### Classifier gate — REVISED with corrected data

| Bucket | OLD PF (5/22) | NEW PF (5/23) | Decision |
|---|---:|---:|---|
| <0.25 (strong_bearish) | 3.64 | 3.64 | **NO VETO** — top winners live here (RKLB, LUNR, SEDG) |
| 0.25-0.35 (bearish) | 0.85 | **1.03** | **NO VETO** — break-even at N=5 is noise, not a loser |
| 0.35-0.40 (neutral) | 2.36 | 1.83 | **NO VETO** — still profitable; news strategy alpha |

> **Chop-zone veto was attempted (commit reverted same day).** Corrected
> data showed PF 1.03 (break-even), not 0.85 (loser). Blocking break-even
> trades has no expected value and limits learning. Classifier remains in
> shadow-mode logging only. **Do NOT use classifier as any kind of long
> gate at this sample size.**

### Short-side branch

- Short counterfactual: 20 trades, +$143, PF **1.10**, WR 40%.
- **DEFER more firmly.** With more samples (14→20), PF dropped from 1.25 to 1.10.
- The "short edge" is shrinking, not growing — likely it was small-sample noise.
- Keep side_classifier in shadow mode. Re-evaluate at N≥40 (doubled threshold).

### Symbol class gates

- **leveraged_etf: BLOCKLIST stays.** 1 trade (SOXS) → -$228, WR 0%.
- mid_cap (N=2) and small_or_other (N=20, PF 1.99) — no new gates.

### Operational integrity flags (still to action)

1. **35% manual-exit rate (8/23 trades)** — bot's exit edge is underrepresented.
2. **ASTS held 1019.8 min overnight** — news strategy was supposed to be intraday.
3. **RKLB closed in 1.3 min for +$300** — target sizing may be too tight.
4. **Token TTL silently rotates** — needs pre-warning at startup (Task #9).

### Pre-Tuesday implementation status

- [x] Disable breakout strategy in `Config().yaml`
- [x] Add 9:35 ET hard block in `core/engine.py`
- [x] Halve `mean_reversion` size via per-strategy `strategy_size_multipliers`
- [x] Fix logging gap (`research/weekly_report.py`: 30s join + reason fallback + normalize)
- [x] ~~Add classifier chop-zone veto~~ → REVERTED (data didn't support)
- [ ] Investigate ASTS overnight hold (Sunday afternoon)
- [ ] Walk-forward backtest news strategy (Sunday afternoon — **gates Tuesday**)
- [ ] Investigate manual-exit rate (Sunday afternoon)
- [ ] Real circuit breakers + P&L dashboard + token TTL warning (Monday)
- [ ] Write reproducible `./deploy.sh` (Monday)
