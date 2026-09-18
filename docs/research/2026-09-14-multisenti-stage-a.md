# Multi-sentiment Stage A — event→theme→action measurement plan
**Owner:** Research | **Date:** 2026-09-14 | **Hari lock:** multi-sentiment autonomy (Fed, earnings, news, CEO/AI shocks) + continuous open-trade monitoring  
**Constraint:** Do NOT loosen locked news Stage A/B floors. **Never auto-exit** permanent hands-off: **MU, HQGE, SPCX** (SNAP toggleable). Theme desk may **alert / recommend**; bot actions only on `managed_by_bot=True` non-hands-off names.  
**Scorecard:** LIVE Schwab fills + shadow theme actions; not vibes.

---

## 1) Event taxonomy (worth measuring)

| Class | Examples | Typical hold / action window | Primary bot action family |
|-------|----------|------------------------------|---------------------------|
| **A. Scheduled macro** | FOMC decision/presser, CPI/NFP, Fed speak | Minutes–hours around print; often **blackout/hard-skip entries** | hard-skip new entries; optional size-down open risk |
| **B. Scheduled micro** | Earnings (own / peer / supplier) | T−1d briefing → T+15m–1d drift (PEAD lane) | thesis exit / size / skip new |
| **C. Unscheduled CEO / lab / policy** | Anthropic/OpenAI “slowdown”, regulation, guidance | Gap + first 15–60m RTH | **theme hard-skip / size-down / shadow exit** |
| **D. Sector theme shock** | AI semis, mega-cap AI, memory HBM, crypto beta | Same session cluster | basket-wide shadow action |
| **E. Single-name idiosyncratic** | CEO resign, fraud, FDA | Minutes | ticker-local only |

**Out of taxonomy (graveyard):** LLM-from-headline trades; news **latency arb** vs UW/Benzinga; sentiment without `news_age_sec` gate.

---

## 2) Lead-time vs free RSS / NewsBus

### Measurement defs (reuse locked news latency)
- `news_age_sec` = signal_ts − source_published_ts  
- `bus_lag_sec` = bus_ingest_ts − source_published_ts  
- `decision_lag_sec` = action_ts − bus_ingest_ts  
- Forward returns at **T+1m / 5m / 15m / 60m** from **first actionable bus event** (not from article prose time alone)

### Skeptical prior (measure, don’t assume edge)
| Event class | Can free RSS / NewsBus beat the move? | Implication |
|-------------|----------------------------------------|-------------|
| FOMC/CPI | Rarely — wire/TV first; calendar blackout is the win | Prefer **hard-skip entries**, not fade/chase |
| Earnings | Mixed — street often priced; PEAD is **days**, not seconds | Overnight-flag PEAD lane; not day-trade latency |
| CEO/lab weekend essay → Monday open | **Essay can land Sat/Sun; dump at RTH open** | Edge is **pre-open theme map + open hard-skip/size**, not mid-morning chase |
| Intraday CEO tweet | Usually late vs HF/Twitter firehose | Shadow only until lead-time proves p50 `news_age_sec` ≪ move half-life |

**Paid feed rule (locked):** if paid does not beat free RSS median by **≥30s** on shared stories → don’t pay (from news Stage A brief).

### First observability cut — Anthropic slowdown (2026-09-14)
**Sourced:** Amodei essay/X ~**Sat 2026-09-13**; Altman/Musk concur weekend; **Mon 2026-09-14** AI/semis slide (CNBC/Reuters/Morningstar): MU/INTC/AMD ~**−5–6%**, NVDA ~**−2–3%**, SOXX weak.  
**Research implication:** this is a **weekend→open theme** event. Correct bot behavior = detect theme Sat–Sun → Mon **hard-skip new AI-basket entries** + **shadow size-down / theme-exit on bot-managed** AI names.  
**MU:** permanent hands-off → **Decision Card / alert only**, never bot exit. Same HQGE/SPCX.

**Lead-time homework (Engine/NewsBus logs next 10 sessions):** for each class-C/D event, log whether `bus_ingest_ts` was **before RTH open**, **first 5m**, or **after 50% of T+60m move**. If &gt;70% of events arrive after half-move → theme desk is **risk filter**, not alpha.

---

## 3) Theme map (event tags → baskets) — no hands-off auto-exits

### Config shape (modular)
```
theme_id: ai_frontier_slowdown
match: [anthropic, amodei, "pace the frontier", openai slowdown, ...]
basket: [NVDA, AMD, AVGO, SMCI, ARM, TSM, ASML, ...]   # bot-tradable
watch_only: [MU, HQGE, SPCX]   # alert/size-recommend UI only; NEVER order
actions_shadow: [hard_skip_entries, size_down_open, thesis_exit]
```

### Rules
1. **Hands-off forever** on MU/HQGE/SPCX: theme match → `action=alert_only` (+ optional operator toggle UI).  
2. SNAP: only if `managed_by_bot` and not Long-Term flag.  
3. Baskets are **data/config**, not hardcoded in exit path — Engine loads YAML/JSON; Research owns measurement tags.  
4. One event → many tickers; dedupe by `event_id` + `theme_id` + symbol (15m).

### Starter themes (v0)
| theme_id | Basket seed (tradable) | watch_only |
|----------|------------------------|------------|
| `ai_compute` | NVDA AMD AVGO SMCI ARM | MU |
| `ai_mega_cap` | MSFT AMZN GOOGL META | — |
| `memory_hbm` | — | MU (+ peers if added tradable) |
| `fed_risk_off` | beta proxies / high-beta desk names | — |
| `earnings_chip` | peer set around reporter | MU if peer |

---

## 4) Promotion gates — shadow → LIVE

Align to **locked news Stage A** (not softer). Separate books per **action type**.

### Stage A — promote **hard-skip new entries** on theme (easiest / safest)
| Metric | Floor |
|--------|------:|
| n theme events | ≥30 (or ≥10 sessions with ≥1) |
| Precision | ≥55%: skip day shows **less loss** than matched no-skip control on basket (or forward ret of skipped candidates ≤0 at T+15m when direction=defensive) |
| False-skip rate | ≤40% (skipped days that were fine / chopped) — report; tighten later |
| Lead-time | ≥50% of skips have `bus_ingest` before 50% of T+60m move |
| Hands-off audit | **0** bot orders on MU/HQGE/SPCX |

### Stage A — promote **size-down open** (bot-managed only)
Same floors as momentum size overlays: PF/expectancy on **shadow size-down vs full-size counterfactual** ≥ locked Δ; max losing day ≤2R on theme days; n≥80 theme-touched position-days.

### Stage A — promote **theme thesis-exit** (bot-managed only)
Reuse news verifier Stage A: n≥80, PF≥1.30, precision≥55%, exp≥+0.05R, DD≤8% on exit book; **plus** news_age cap (120s intraday / measure first).  
**Never** applies to hands-off names.

### Stage B / LIVE capital
News Stage B floors; first LIVE theme actions ≤1–2% risk capital; Hari opt-in; day-trade LIVE entries remain subject to separate momentum NO_GO until that Stage A greens.

**Default:** shadow-only (`ENABLE_ACTIVE_OPEN_DESK` / theme actions) until Stage A green. No LLM autopilot.

---

## 5) Observability / first “backtest” cut (do now)

### Log schema (Engine)
`event_id, theme_id, source, source_published_ts, bus_ingest_ts, news_age_sec, symbols_tradable[], symbols_watch_only[], action_shadow, managed_flags{}, forward_ret_{1,5,15,60}m, half_move_before_ingest bool`

### First cut status (2026-09-14)
| Item | Status |
|------|--------|
| Anthropic→AI semis Mon dump | **Observed in public tape** (MU/INTC ~5–6%, NVDA ~2–3%) — validates theme class C/D |
| NewsBus lead-time on this event | **TBD** — need KiddoKingdom ingest timestamps vs Sat essay / Mon open |
| Shadow theme actions | **Not yet scored** — enable shadow logger first |
| Hands-off | MU correctly **must not** be bot-exited on this theme |

### Minimal Engine sequence (for CoS)
1. Theme YAML + NewsBus tagger (shadow log only)  
2. `watch_only` path for MU/HQGE/SPCX  
3. Research weekly Stage A rollup on theme hard-skip book  
4. Only then size-down / thesis-exit shadows  

---

## One-screen ask for Hari/CoS
Approve taxonomy A–E + Stage A gates above? Confirm **hard-skip entries** is the first LIVE candidate (not theme-exit on opens). MU/HQGE/SPCX stay alert-only forever.

---

## Appendix — lead-time evidence cut (2026-09-14)

| Finding | Source |
|---------|--------|
| Equity index futures reprice macro surprises in ~**5 ms** | Chordia/Green/Kottimukkalur, RFS 2018 |
| FOMC: pressers now dominate vs statements | FRBSF WP 2025-30 Acosta et al. |
| After-hours earnings discovery ms-scale; delays/spreads kill sim profits | arXiv:2601.08962; Grégoire & Martineau JAR 2022 |
| Free RSS lag often cited 5–30 min | vendor/practitioner — **unverified** |
| Anthropic Amodei essay **Sat 2026-09-12** → U.S. session **Mon 2026-09-14** | Reuters/CNBC/NBC |
| Same-day: SOX **−5.1%**, NVDA **−3.6%**, AMD **−5.6%**, MU **−6%**, NDX **−1.2%** (Reuters early); indexes recovered off lows ~2pm ET | Reuters/NBC |

**Stage A implication unchanged:** measure residual T+5m/15m/1h and open hard-skip quality — do not claim wire races. Full notes: `2026-09-14-multisenti-leadtime-notes.md`.
