# Stage A — GPU news sense-making vs keyword ThemeShock
**Owner:** Research | **Date:** 2026-09-14 | **Hari lock:** RTX 3090 digs many reliable sources + *understands* (not keyword FP)  
**Prior art:** locked news Stage A/B; multi-sentiment ThemeShock plan; Crude Oil summary contamination RCA  
**Constraint:** No LLM-from-headlines *orders*. GPU = scorer/critic/embeddings inside NewsBus + risk rails. Hands-off MU/HQGE/SPCX never bot-exited. Floors not loosened.

---

## 0) Roles (keep both — promote GPU as *filter*, not replacement day 1)

| Layer | Job | Failure mode we already saw |
|-------|-----|------------------------------|
| **Keyword ThemeShock** | Fast recall / candidate gen | Summary contamination (Crude Oil + Anthropic blurb → hard_skip) |
| **GPU sense-maker** | Precision: is this *really* an AI-compute/Fed/earnings *theme event* for these tickers? | Latency, hallucination, over-refusal |
| **Actuator** | Shadow hard_skip / size / thesis_exit → LIVE only after Stage A | Acting on FP keywords |

**Promotion order:** keyword shadow stays on for recall → GPU critic shadows in parallel → LIVE hard_skip only when **GPU-gated** keyword (or GPU-primary) clears Stage A.

---

## 1) What GPU scores (event object)

For each NewsBus item (multi-source: Alpaca + RSS + later paid), GPU emits a structured card (no free-form trade):

```
event_id, sources[], source_published_ts, bus_ingest_ts,
theme_probs{ai_compute, ai_mega_cap, fed_risk_off, earnings_chip, memory_hbm, other, none},
relevance_by_symbol{SYM: 0–1},
stance{bullish,bearish,mixed,irrelevant},
contamination_risk{0–1},   # multi-story / off-topic summary
confidence{0–1},
rationale_short,            # critic text, not an order
keyword_match{theme_id, matched_text, matched_field} | null
```

**Contamination filter (Crude Oil case):** if `contamination_risk ≥ 0.6` OR `matched_field=summary` with low headline theme_prob → **suppress hard_skip** (allow alert_only). Require `theme_prob[theme] ≥ τ` **and** `relevance[symbol] ≥ ρ` before hard_skip shadow.

---

## 2) Lead-time measurement (same defs)

- `news_age_sec`, `bus_lag_sec`, `gpu_infer_ms`, `decision_lag_sec`
- Forward ret T+1/5/15/60m from first *actionable* bus event
- **SLO (shadow):** p95 `gpu_infer_ms` &lt; **800ms** on 3090 for card; e2e decision &lt; **2s** after ingest for hard_skip shadow
- Skeptical prior: GPU won’t beat wires on FOMC ms; wins on **weekend→open theme understanding** + **FP rejection**

---

## 3) Promotion gates — three books

### A) Keyword ThemeShock alone (baseline — already collecting)
Use multi-sentiment Stage A (hard_skip first). Track FP rate (Crude Oil-class).

### B) GPU critic on keyword candidates (first LIVE candidate)
Promote **GPU-gated hard_skip** when all green on rolling window:

| Metric | Floor |
|--------|------:|
| n gated decisions | ≥80 (or ≥10 sessions) |
| Precision vs human/label or vs post-hoc move | ≥**65%** (stricter than keyword 55% — understanding bar) |
| FP rate on contamination set | ≤**5%** (Crude Oil-class must not hard_skip) |
| Recall vs keyword true positives (Anthropic-class) | ≥**80%** (don’t kill true AI themes) |
| Lead-time | ≥50% of gated skips before half of T+60m move **or** pre-open for overnight essays |
| Hands-off | **0** bot orders on MU/HQGE/SPCX |
| Latency | p95 gpu_infer_ms &lt; 800; missing-score rate &lt; 2% |

### C) GPU-primary theme (no keyword required) — Stage B later
Only after B green; n≥150; same precision/FP; prove incremental lift over B (Δ precision ≥ +5pp or Δ $/day &gt; 0 on skipped book).

---

## 4) Label / observability cut (first week)

1. Log every keyword hard_skip with `matched_text` + `matched_field` (Engine precision PR).  
2. Shadow GPU score on same items (offline batch OK first on 3090).  
3. Build contamination golden set: Crude Oil event_id `1a1e82cd9500` + 20 similar multi-story Alpaca items; Anthropic true set ≥20.  
4. Report weekly: keyword FP%, GPU-gated FP%, recall on Anthropic set, latency.

**Success for Hari:** GPU rejects Crude Oil-class; keeps Anthropic→ai_compute; MU stays watch_only/alert.

---

## 5) Engine coordination (after their inventory)

Ask Engine for:
- Where 3090 inference will live (local service vs in-process)
- Multi-source dig list (Alpaca already; RSS; which next)
- Hook: NewsBus.on_publish → GPU scorer → ThemeShock gate
- Config flags: `ENABLE_GPU_NEWS_CRITIC` (shadow), `THEME_HARD_SKIP_REQUIRE_GPU` (default False until Stage A)

Research owns: gates, golden sets, weekly rollup (extend ThemeShock Monday routine).

---

## One-screen ask
Approve **GPU-as-critic gating keyword hard_skip** as first LIVE path (book B), not GPU-primary day 1? Floors above OK?
