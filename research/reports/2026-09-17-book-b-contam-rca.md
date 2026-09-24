# Book B contamination FP RCA — 3 ai_compute Corning/oil leaks

**As of:** 2026-09-17T16:34:35-04:00 (ET)  
**Audience:** CoS + Engine  
**Seed:** `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json` (tip `f6fb73c`)  
**Scorecard:** `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard.md`  
**Policy:** `/workspace/briefs/2026-09-14-gpu-sense-stage-a.md`  
**Critic code:** `analysis/gpu_news_critic.py` (`GPUNewsCriticModel._compute_contamination_risk`, `THEME_DESCRIPTIONS`)

    10|**Locks (unchanged):** `THEME_HARD_SKIP_REQUIRE_GPU=False` · LIVE hard_skip=**NO** · no LIVE / no require_gpu flip in this brief.

---

## Problem

Book B Stage A **contam_fp = 15.0% (3/20) FAIL** vs floor ≤5%. Precision 85.7% PASS and anthropic/ai_recall 90% PASS, but overall gate stays **INCOMPLETE** (n=40 <80, sessions=n/a) and contamination blocks any promote path.

Book B rule (scorecard): keyword candidate ∧ critic present ∧ NOT suppress. Suppress = would_suppress* / contam≥0.6 / summary+non-align / **top_theme ∉ {ai_compute, ai_mega_cap, memory_hbm}**. On this seed every critic `action` is `shadow_pass` (0× would_suppress), so the gate is effectively **top_theme ∈ AI_ALIGN** + contam/summary. The 3 leaks all have `top_theme=ai_compute` with `contamination_risk=0.0`.

    20|---

## 1) Exact 3 FP nests (from seed — never fabricated)

| # | event_id | headline | label | expected |
|--:|----------|----------|-------|----------|
| 1 | `f59b23116fcb` | Why Corning Plunged Today | FP_contamination_critic_wrong_theme | no_hard_skip |
| 2 | `9579cf0d1b4f` | The Oil Crisis Has Reached Costco's Motor Oil Aisle | FP_contamination_critic_wrong_theme | no_hard_skip |
| 3 | `2d83a5133335` | Corning Rides on Expanding Partner Base: Will it Boost Prospects? | FP_contamination_critic_wrong_theme | no_hard_skip |

    30|### Shared critic pattern

All three: `action=shadow_pass`, `contamination_risk=0.0`, `keyword_match=null`, row-level `matched_field`/`matched_text` absent, `full_theme_probs=true`, source=`bot_decisions`, join=`event_id`. Softmax is **near-uniform** (7-way ≈0.143 baseline); `ai_compute` wins by a thin tip.

### Nest A — `f59b23116fcb`

```
top_theme=ai_compute  top_theme_prob=0.16060430533665335
theme_probs={
  ai_compute: 0.16060430533665335,   # winner
    40|  earnings_chip: 0.14675367420594612,
  ai_mega_cap: 0.14629902712974507,
  none: 0.14059466984574645,
  other: 0.1405928050181133,
  fed_risk_off: 0.13293979382469187,
  memory_hbm: 0.13221572463910386
}
margin(top−2nd)=0.01385
contamination_risk=0.0  action=shadow_pass  stance=irrelevant  confidence=0.402
keyword_match=null  symbols_seen=[GLW]
    50|relevance_by_symbol={AMD,ARM,AVGO,NVDA,SMCI ≈0.128 each}  # basket priors, not GLW
infer_ms=6.32  attached_from_tip=a17bb6b
```

### Nest B — `9579cf0d1b4f`

```
top_theme=ai_compute  top_theme_prob=0.15553225470761348
theme_probs={
  ai_compute: 0.15553225470761348,   # winner
    60|  memory_hbm: 0.14722036061216273,
  fed_risk_off: 0.14314581107120192,
  none: 0.13942549833529164,
  other: 0.1392157034823207,
  earnings_chip: 0.138668406892495,
  ai_mega_cap: 0.13679196489891454
}
margin=0.00831
contamination_risk=0.0  action=shadow_pass  stance=irrelevant  confidence=0.352
keyword_match=null  symbols_seen=[NVDA]
    70|relevance_by_symbol={AMD,ARM,AVGO,NVDA,SMCI ≈0.124 each}
infer_ms=27.44  attached_from_tip=a17bb6b  soak_ts on row
```

### Nest C — `2d83a5133335` (soak 2026-09-17)

```
top_theme=ai_compute  top_theme_prob=0.16164554734498243
theme_probs={
  ai_compute: 0.16164554734498243,   # winner by hair
    80|  ai_mega_cap: 0.16122925464489538,  # nearly tied
  earnings_chip: 0.14981737301333242,
  fed_risk_off: 0.1331624198364432,
  none: 0.1331470615250279,
  memory_hbm: 0.13265125778162606,
  other: 0.12834708585369264
}
margin=0.000416   # effectively a coin-flip vs ai_mega_cap
contamination_risk=0.0  action=shadow_pass  stance=bullish  confidence=0.383
keyword_match=null  symbols_seen=[META]
    90|relevance_by_symbol={AMD,ARM,AVGO,NVDA,SMCI ≈0.129 each}
infer_ms=14.99  attached_from_tip=ad05693
```

**Contrast (suppressing FPs on same seed):** 17/20 other FPs land `top_theme=earnings_chip` (~0.15–0.17) → Book B correctly suppresses via AI_ALIGN miss. Contam still 0.0 on almost all of them — lexical contam is **not** what is saving us today; accidental earnings_chip tip is.

---

## 2) RCA — why critic ranked `ai_compute` over earnings_chip / oil / etc.

   100|### Mechanism

1. **Zero-shot MiniLM vs short prototypes** (`THEME_DESCRIPTIONS`). Live path normalizes cosine sims into a 7-way softmax. Short, vague headlines ("Why Corning Plunged Today") produce a **flat distribution** (~0.13–0.16). Argmax is noise-dominated; a few points of embedding drift flip the Book B gate.

2. **Corning ↔ AI optical narrative bleed.** Market discourse links GLW/Corning to AI datacenter fiber. Prototype text for `ai_compute` ("AI compute infrastructure… data center capacity") pulls Corning headlines toward `ai_compute` even when the story is equity offering / plunge / partner-base — **not** a GPU/HBM theme event. Sibling Corning FPs often tip `earnings_chip` instead; these three tipped the other way.

3. **Oil aisle wrap has no oil-theme class.** There is no `commodity_oil` / `energy` theme. "Oil Crisis… Motor Oil Aisle" cannot land on a dedicated class; FakeCritic triggers (`crude oil`, `petroleum`, `opec`, `oil prices`, `wti`/`brent`) do **not** match "motor oil" / "oil crisis". Live `_compute_contamination_risk` has **no lexical commodity/glass branch** — only summary-kw low-prob, other+specific, long-summary, kw-theme disagreement. With `keyword_match=null` and short headlines → **risk stays 0.0**.

4. **Symbol path ≠ relevance path.** `symbols_seen` is GLW / NVDA / META (ingest/fanout), but `relevance_by_symbol` is a near-uniform dump of the **ai_compute basket** (AMD/ARM/AVGO/NVDA/SMCI ≈0.12). Critic never scores GLW relevance; Book B never checks `relevance[symbol] ≥ ρ` (Stage A §1 mentions ρ but τ/ρ unset — scorecard uses top_theme AI_ALIGN proxy).

   110|5. **`would_suppress` never fires.** Suppress path needs `contamination_risk ≥ 0.6` or summary-kw low theme_prob. Contam=0 and kw=null → `shadow_pass` → Book B hard_skips purely because argmax ∈ AI_ALIGN.

6. **Margin / τ alone cannot separate FP from TP on current model.** AI_ALIGN TPs on seed also sit at top_prob ≈0.15–0.18 with margins 0.0004–0.019 (all <0.02). A naive `top_theme_prob ≥ 0.20` or margin gate zeros **both** contam FPs **and** TP recall. Confirm-τ needs a sharper model or a non-prob signal (lexicon / headline kw / basket).

### Why Book B still hard_skips them

| Gate clause | Result on 3 FPs |
|-------------|-----------------|
| Keyword ThemeShock candidate (seed membership) | yes (Book A FP set) |
| gpu_critic present | yes |
   120|| action would_suppress* | no (`shadow_pass`) |
| contam ≥ 0.6 | no (0.0) |
| summary + non-align | n/a (`keyword_match=null`) |
| top_theme ∈ AI_ALIGN | **yes → hard_skip LEAK** |

---

## 3) Concrete Engine fix proposal (modular, config flags, safe off-path)

**Do not flip `THEME_HARD_SKIP_REQUIRE_GPU`.** Keep shadow-only. Prefer extractable hooks in `gpu_news_critic.py` + `Config` properties (Hari modular rule).
   130|
### Recommended primary (best counterfactual on this seed)

**A. Raise `contamination_risk` on commodity + glass peer wraps** — config lexicon, default **off** or shadow-scoring only until soak.

| Flag | Default | Behavior |
|------|---------|----------|
| `GPU_CRITIC_CONTAM_LEXICON_ENABLE` | **False** (safe off-path) | When True, apply expanded triggers inside `_compute_contamination_risk` |
| `GPU_CRITIC_CONTAM_LEXICON` | list (config/YAML) | Add: `corning`, ` glw`, `nyse:glw`, `motor oil`, `oil crisis`, `crude oil`, `petroleum`, `opec`, `wti`, `brent`, peer tokens `cohr`/`cien`/`aaoi` when co-mentioned with glass/optical without AI headline triggers |
| bump | `max(risk, 0.65)` on headline|summary hit **unless** headline also has AI_COMPUTE_TRIGGERS (`anthropic`, `openai`, `gpu demand`, `data center`, …) | Forces `would_suppress_hard_skip` under existing ≥0.6 rule |
   140|
**Seed counterfactual:** contam_fp **0/20 (0%)**, TP retain **17/20 (85%)** — still ≥80% recall. The one TP touched (`bf0c664141dc` Nasdaq/AI Leaders…) is a multi-ETF wrap that lists GLW in `symbols_seen`; refine lexicon to **headline/summary text** (not symbols_seen dump) to keep that TP.

Also mirror lexicon into `FakeCritic.CONTAMINATION_TRIGGERS` so CI doubles stay honest.

### Secondary options (compose behind flags)

| # | Flag / hook | Intent | Seed counterfactual | Caution |
|---|-------------|--------|---------------------|---------|
| **B** | `GPU_CRITIC_HARD_SKIP_REQUIRE_HEADLINE_KW` (shadow Book-B scorer only; default False) | hard_skip only if `keyword_match.matched_field==headline` ∧ top ∈ AI_ALIGN | FP 0%; TP retain **7/20 (35%)** — **recall FAIL** | Too strict until critic kw attach coverage rises; OK as *measurement* gate, not LIVE |
   150|| **C** | `GPU_THEME_CONFIRM_TAU` (default unset/0) | require `top_theme_prob ≥ τ` | τ=0.20 → FP 0% but TP **0%** on current MiniLM flat softmax | **Do not enable** until model calibration; document τ as future confirm gate per Stage A §1 |
| **D** | `GPU_CRITIC_EXCLUDE_BASKET` / headline denylist (Corning/oil) | suppress hard_skip when headline matches glass/oil denylist | FP 0%; TP 17/20 | Overlaps A; prefer A (contam→would_suppress) so observability stays on `contamination_risk` |
| **E** | `relevance[symbol] ≥ ρ` before hard_skip | Stage A §1 unfinished | Would reject GLW/META/Costco paths where relevance is only basket-uniform | Needs real per-symbol relevance, not basket prior dump |
| **F** | theme_probs margin threshold | reject flat argmax | Does **not** separate FP vs TP on this seed (all TP margins <0.02) | Skip as sole gate |

### Preferred ship order for Engine

1. **PR1 (shadow):** Lexicon A behind `GPU_CRITIC_CONTAM_LEXICON_ENABLE=0` default; unit tests on the 3 eids → `contamination_risk≥0.6` + `would_suppress_hard_skip`; Anthropic headline TPs unchanged.  
2. **PR2 (observability):** Persist `matched_field` from ThemeShock into critic card when present; log lexicon hit reason in `rationale_short`.  
3. **PR3 (research soak):** Enable lexicon in shadow only on KiddoKingdom; re-score Book B scorecard; **still** leave `THEME_HARD_SKIP_REQUIRE_GPU=False`.  
   160|4. Defer B/C until attach+calibration support ≥80% recall.

### One-paragraph Engine ask

Please add a **config-gated contamination lexicon** in `GPUNewsCriticModel._compute_contamination_risk` (and mirror in `FakeCritic`) that bumps `contamination_risk` to ≥0.65 on Corning/GLW/motor-oil/oil-crisis/crude commodity wraps unless an AI headline trigger is also present — default **off**, shadow-only, no `THEME_HARD_SKIP_REQUIRE_GPU` flip — so the existing ≥0.6 would_suppress path kills the three Book B leaks (`f59b23116fcb`, `9579cf0d1b4f`, `2d83a5133335`) without rewriting the hard_skip actuator; optionally follow with headline-kw / τ / relevance-ρ flags as separate modular off-path knobs once calibration allows.

---

## 4) Labeled-n growth plan → n≥80 / ≥10 sessions (no fabricated labels)

   170|Current: **n_labeled=40** (20 FP / 20 TP), sessions=**n/a**. Need +≥40 labeled **or** session accounting ≥10. Do not invent nests or theme_probs.

### Sources (honest mine only)

| Source | How | Notes |
|--------|-----|-------|
| ThemeShock emits | Continue KiddoKingdom soak; mine `theme_shock_logger` / hard_skip_entries + alert_only | Post-#56 headline-only → oil∩ai_compute ThemeShock ≈0; expect slow FP growth from keyword path |
| Critic DB | `rudra_dev.bot_decisions` component=`gpu_news_critic` (already 596+ unique in soaks) | Prefer full `theme_probs` attach; never fabricate; join event_id then headline |
| Logs | `trading_bot.log[.1–.5]`, `/tmp/trading_bot_stdout.log` | Same soak scripts as 2026-09-15/16/17 deltas |
| Session counter | Define session = distinct RTH date with ≥1 ThemeShock or critic emit on tip | Backfill from log/DB dates → unlock ≥10 sessions even before n=80 |
   180|
### Label criteria (locked — do not stretch)

- **FP:** Crude Oil / Corning / commodity multi-story or critic-wrong-theme (`top=ai_compute` on glass/oil without AI primary). `expected_action=no_hard_skip`. Reject pure Fed/oil wraps with no Corning/ai_compute wrong-theme (see soak 2026-09-17 rejects).  
- **TP:** Anthropic / genuine AI-compute headline hits. Prefer `matched_field=headline`. `expected_action=hard_skip_entries`.  
- **Hands-off:** MU / HQGE / SPCX — watch_only only; no invented LIVE hard_skip.  
- **gpu_critic:** attach only from logs/DB; partial OK if marked; **never fabricate**.

### Growth tactics

   190|1. **Daily soak delta** (existing `book_a_fp_soak_*` pattern): cut timestamp → mine → append only new event_ids → backup seed → brief. Target +2–5 FP/week if lexicon not yet live; more once critic wrong-theme surface expands.  
2. **Re-attach missing TPs:** `63251da54499`, `8e3bb2b090ef` (no_attach today) — DB/log join only; lifts recall denom quality.  
3. **TP growth:** mine critic/ThemeShock for Anthropic / AI-slowdown / NVDA-compute headlines with durable event_id; keep label bar high.  
4. **Session ledger:** add `sessions` field to seed/scorecard JSON from distinct RTH dates in emits (n/a → real count).  
5. **Balance:** aim ~1:1 FP:TP toward 40/40 (n=80); do not pad FPs with non-contamination classes (prior drops: NIO/UiPath/HPE/PLTR/Oracle stretch).  
6. **After Engine lexicon PR shadows on:** re-score Book B; expect contam_fp → ≤5% on current 20; still need n/sessions for GO.

### Soak exit criteria (promote readiness — not this brief)

- n_labeled ≥80 **or** sessions ≥10  
   200|- contam_fp ≤5% under Book B rule (post-lexicon)  
- precision ≥65%, anthropic/ai_recall ≥80%, p95 infer &lt;800ms  
- `THEME_HARD_SKIP_REQUIRE_GPU` still False until CoS explicit promote  
- LIVE hard_skip = **NO** until all green

---

## CoS crisp line

`Book B contam RCA: 3/20 FPs (f59b23116fcb, 9579cf0d1b4f, 2d83a5133335) leak because MiniLM flat softmax tips ai_compute on Corning/oil wraps with contam=0/kw=null; ask Engine config-gated contam lexicon (default off) → would_suppress; grow n via ThemeShock+critic DB soak to ≥80/≥10 sessions; LIVE/require_gpu unchanged.`
   210|
---

## Paths

- This RCA: `/workspace/briefs/2026-09-17-book-b-contam-rca.md`  
- Scorecard: `/workspace/briefs/2026-09-17-book-b-stage-a-scorecard.md` (+ `.json`)  
- Seed: `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json`  
- Critic: `kiddo_stage_a/repo/analysis/gpu_news_critic.py`
