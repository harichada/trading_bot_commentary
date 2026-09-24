# Book B Stage A Scorecard — ThemeShock/GPU golden set

**As of:** 2026-09-17T16:32:41-04:00 (ET)  
**Seed:** `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json` (tip `f6fb73c`, seed as_of `2026-09-14T16:05:15.089155-04:00`)  
**Policy:** `/workspace/briefs/2026-09-14-gpu-sense-stage-a.md`  
**Book A:** keyword ThemeShock baseline (`expected_action` on labels)  
**LIVE hard_skip rec:** **NO** (locked; do not promote)

---

## Exact Book B decision rule

**Book B `hard_skip` IFF all of:**

1. **Keyword ThemeShock candidate** — golden-set row is on the Book A keyword path (FP/TP seed membership).
2. **`gpu_critic` nest present** — use attached fields only; **never fabricate**.
3. **Critic does NOT suppress**, where suppress = ANY of:
   - `action` ∈ {shadow_would_suppress, would_suppress_hard_skip, would_suppress, shadow_would_suppress_hard_skip}
   - `contamination_risk ≥ 0.6` (Stage A §1 Crude Oil filter)
   - `matched_field=summary` AND `top_theme` ∉ AI_ALIGN
   - **`top_theme` ∉ AI_ALIGN** — proxy for unset Stage A `theme_prob[theme] ≥ τ` confirm gate

**AI_ALIGN** = `{ai_compute, ai_mega_cap, memory_hbm}` (ai_compute keyword family; keeps Anthropic TPs whose top is `ai_mega_cap`).

**Missing critic** → `no_attach` (not a Book B hard_skip; counts against Anthropic/ai_compute recall).

**Note:** On this seed every attached critic `action` is `shadow_pass` (0× would_suppress). Gate therefore hinges on **top_theme AI_ALIGN** + contam/summary filters.

---

## Counts

| | n |
|--|--:|
| n_fp | **20** |
| n_tp | **20** |
| n_labeled | **40** |
| sessions | **n/a** (not in seed) |
| attach any | **38/40 (95.0%)** — FP 20/20, TP 18/20 |
| attach full (multi-key theme_probs / no `_partial`) | **35/40 (87.5%)** — FP 19/20, TP 16/20 |
| Book B hard_skip | **21** (TP 18 + FP 3) |

GO-bar seed size: FP≈20 / TP≈20 — **hit**. Promote n-floor (≥80 labeled or ≥10 sessions) — **not hit**.

---

## Metrics vs locked gates

| Gate | Floor | Value | Status |
|------|------:|------:|:------:|
| n labeled / sessions | ≥80 **or** ≥10 sessions | n=40, sessions=n/a | **INCOMPLETE** |
| Precision | ≥65% | **85.7%** (18/21) | **PASS** |
| Contamination FP rate | ≤5% | **15.0%** (3/20) | **FAIL** |
| Anthropic/ai_compute recall | ≥80% | **90.0%** (18/20) | **PASS** |
| p95 `gpu_critic.infer_ms` | <800ms | **19.70ms** (p50=7.72, n_infer=38) | **PASS** |

**Anthropic-headline subset (sanity):** 6/6 = 100.0% retained (all 6 Anthropic TPs have critic + AI_ALIGN top).

### FP still hard_skip under Book B (contamination leaks)

| event_id | top_theme | label | headline |
|----------|-----------|-------|----------|
| `f59b23116fcb` | ai_compute(0.161) | FP_contamination_critic_wrong_theme | Why Corning Plunged Today |
| `9579cf0d1b4f` | ai_compute(0.156) | FP_contamination_critic_wrong_theme | The Oil Crisis Has Reached Costco’s Motor Oil Aisle |
| `2d83a5133335` | ai_compute(0.162) | FP_contamination_critic_wrong_theme | Corning Rides on Expanding Partner Base: Will it Boost Prospects? |

### TP not retained

| event_id | book_b | reason | headline |
|----------|--------|--------|----------|
| `63251da54499` | no_attach | missing_gpu_critic | AI stocks fall after Amodei, Altman, and Musk back AI slowdo |
| `8e3bb2b090ef` | no_attach | missing_gpu_critic | Why Is Nvidia (NASDAQ:NVDA) Stock Falling Today? NVDA Drops |

---

## Overall

| | |
|--|--|
| **Overall gate** | **INCOMPLETE** (n&lt;80 and no session count; contamination also FAIL) |
| **Promote Book B LIVE hard_skip?** | **NO** |
| **THEME_HARD_SKIP_REQUIRE_GPU** | leave **False** |

### Research → CoS crisp line

`Book B Stage A: n=40 INCOMPLETE; precision=85.7% PASS; contam_fp=15.0% FAIL (3/20 top=ai_compute Corning/oil); anthropic/ai_recall=90.0% PASS; p95_infer=19.7ms PASS; LIVE hard_skip=NO.`

---

## Paths

- Scorecard MD: `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard.md`
- Scorecard JSON: `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard.json`
- Seed: `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json`
