# Book B Stage A Scorecard r2 — lexicon-ON counterfactual (post #93)

**As of:** 2026-09-17T16:46:49-04:00 (ET)  
**Seed:** `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json` (tip `f6fb73c`, seed as_of `2026-09-14T16:05:15.089155-04:00`)  
**Prior scorecard:** `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard.md`  
**RCA:** `/workspace/briefs/2026-09-17-book-b-contam-rca.md`  
**PR #93:** `GPU_CRITIC_CONTAM_LEXICON_ENABLE` — tip **`38117b9`** (local lexicon commit `f648b0a`)  
**Policy:** `/workspace/briefs/2026-09-14-gpu-sense-stage-a.md`  
**Book A:** keyword ThemeShock baseline (`expected_action` on labels)  
**LIVE hard_skip rec:** **NO** (locked; do not promote)  
**THEME_HARD_SKIP_REQUIRE_GPU:** leave **False** (HOLD)

---

## Counterfactual method (lexicon-ON on stored cards)

**Not a live re-infer.** For each golden-set row, apply PR #93 lexicon rules to `headline` / `summary` / other seed text fields (`matched_text` if present) — **not** `symbols_seen`:

1. If any token in `{corning, glw, nyse:glw, motor oil, oil crisis, crude oil, petroleum, opec, wti, brent, cohr, cien, aaoi}` appears (case-insensitive substring), **and**
2. Headline does **not** contain an `AI_COMPUTE_TRIGGERS` hit (`anthropic`, `amodei`, `openai`, `frontier ai`, `ai slowdown`, `gpu demand`, `data center`/`data-center`/`datacenter`, `ai compute`, `ai infrastructure`),

then treat **effective** `contamination_risk = max(stored, 0.65)` even when stored `gpu_critic.contamination_risk` was `0.0`. That trips the existing Book B suppress clause `contam ≥ 0.6`.

---

## Exact Book B decision rule (unchanged + lexicon CF)

**Book B `hard_skip` IFF all of:**

1. **Keyword ThemeShock candidate** — golden-set row is on the Book A keyword path (FP/TP seed membership).
2. **`gpu_critic` nest present** — use attached fields only; **never fabricate**.
3. **Critic does NOT suppress**, where suppress = ANY of:
   - `action` ∈ {shadow_would_suppress, would_suppress_hard_skip, would_suppress, shadow_would_suppress_hard_skip}
   - `contamination_risk ≥ 0.6` (**effective** = max(stored, 0.65) under lexicon-ON CF above)
   - `matched_field=summary` AND `top_theme` ∉ AI_ALIGN
   - **`top_theme` ∉ AI_ALIGN** — proxy for unset Stage A `theme_prob[theme] ≥ τ` confirm gate

**AI_ALIGN** = `{ai_compute, ai_mega_cap, memory_hbm}`.

**Missing critic** → `no_attach` (not a Book B hard_skip; counts against Anthropic/ai_compute recall).

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
| Book B hard_skip | **18** (TP 18 + FP 0) |
| Lexicon-suppress FP (CF) | **20** (incl. prior 3 leaks) |
| Lexicon-suppress TP (CF) | **0** |

GO-bar seed size: FP≈20 / TP≈20 — **hit**. Promote n-floor (≥80 labeled or ≥10 sessions) — **not hit**.

---

## Metrics vs locked gates

| Gate | Floor | Value | Status |
|------|------:|------:|:------:|
| n labeled / sessions | ≥80 **or** ≥10 sessions | n=40, sessions=n/a | **INCOMPLETE** |
| Precision | ≥65% | **100.0%** (18/18) | **PASS** |
| Contamination FP rate | ≤5% | **0.0%** (0/20) | **PASS** |
| Anthropic/ai_compute recall | ≥80% | **90.0%** (18/20) | **PASS** |
| p95 `gpu_critic.infer_ms` | <800ms | **19.70ms** (p50=7.72, n_infer=38) | **PASS** |

**Before → after (lexicon-OFF r1 → lexicon-ON r2 CF):**

| Metric | r1 (prior) | r2 (this) |
|--------|----------:|----------:|
| contam_fp | **15.0%** (3/20) FAIL | **0.0%** (0/20) PASS |
| precision | 85.7% | 100.0% |
| anthropic/ai_recall | 90.0% | 90.0% |
| n hard_skip | 21 | 18 |

**Anthropic-headline subset (sanity):** 6/6 = 100.0% retained.

### Prior contamination leaks — lexicon-killed (expect 0 remaining)

| event_id | lexicon hits | stored→effective contam | book_b | headline |
|----------|--------------|-------------------------|--------|----------|
| `f59b23116fcb` | corning | 0.0→0.65 | **suppress** | Why Corning Plunged Today |
| `9579cf0d1b4f` | motor oil, oil crisis | 0.0→0.65 | **suppress** | The Oil Crisis Has Reached Costco's Motor Oil Aisle |
| `2d83a5133335` | corning | 0.0→0.65 | **suppress** | Corning Rides on Expanding Partner Base: Will it Boost Prospects? |

### FP still hard_skip under Book B (contamination leaks)

**None** — contam_fp = 0/20.

### TP not retained

| event_id | book_b | reason | headline |
|----------|--------|--------|----------|
| `63251da54499` | no_attach | missing_gpu_critic | AI stocks fall after Amodei, Altman, and Musk back AI slowdo |
| `8e3bb2b090ef` | no_attach | missing_gpu_critic | Why Is Nvidia (NASDAQ:NVDA) Stock Falling Today? NVDA Drops |

### TP lexicon collateral

**None** — no TP lost to lexicon (text-only match; `bf0c664141dc` GLW/COHR only in `symbols_seen`, correctly ignored).

---

## Overall

| | |
|--|--|
| **Overall gate** | **INCOMPLETE** (n&lt;80 and no session count; contam now PASS but n-floor blocks GO) |
| **Promote Book B LIVE hard_skip?** | **NO** |
| **THEME_HARD_SKIP_REQUIRE_GPU** | leave **False** (HOLD) |

### Research → CoS crisp line

`Book B Stage A r2 (lexicon-ON CF tip 38117b9/#93): n=40 INCOMPLETE; precision=100.0% PASS; contam_fp=15.0%→0.0% PASS (0/20; prior leaks f59b/9579/2d83 lexicon-killed); anthropic/ai_recall=90.0% PASS; p95_infer=19.7ms PASS; LIVE hard_skip=NO; HOLD THEME_HARD_SKIP_REQUIRE_GPU.`

---

## Paths

- Scorecard MD: `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard-r2.md`
- Scorecard JSON: `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard-r2.json`
- Prior r1: `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard.md`
- RCA: `/workspace/briefs/2026-09-17-book-b-contam-rca.md`
- Seed: `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json`
