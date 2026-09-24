# Book A FP soak delta — Crude Oil/Corning contamination

**As of:** 2026-09-17T16:28:48.952254-04:00  
**Soak cut:** `2026-09-16T16:52:03-04:00` (after prior soak 2026-09-16)  
**Seed:** `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json`  
**Seed book tip (unchanged):** `f6fb73c`  
**Attached from tip:** `ad05693` (KiddoKingdom HEAD)  

## Before / after

| | Before | After | Δ |
|--|--:|--:|--:|
| FP count | 18 | 20 | +2 |
| TP count | 20 | 20 | 0 |
| FP shortfall vs ~20 | 2 | 0 | -2 |
| Full gpu_critic attach (all) | — | 35/40 (87.5%) | — |
| Full gpu_critic attach (FP) | — | 19/20 (95.0%) | — |

## Mine summary (since soak cut)

| Source | Finding |
|--------|---------|
| theme_shock_logger after cut | 32 unique eids / theme_ids={'ai_mega_cap': 12, 'fed_risk_off': 9, 'ai_compute': 7, 'memory_hbm': 4}; ai_compute=7; **oil/Corning ∩ ai_compute = 0** |
| gpu_news_critic logs after cut | 596 unique eids |
| DB kw=ai_compute ∩ oil/Corning | **0** |
| Honest new FPs labeled | **2** (raw mine FP_ 3; 1 dup `cbfe934f67f5` already in seed) |
| Rejected / not FP | Pure oil/Fed wraps without ai_compute wrong-theme or Corning class (Treasury/oil $100, Triple Witching/Falling Oil, Nasdaq oil slides, Bloom Energy/Falling Oil, Crude Oil Down 1% stretch) |
| Hands-off | MU/HQGE/SPCX — no invented LIVE hard_skip |

## New FPs appended (deduped by event_id)

| event_id | label | top_theme | contam | headline |
|----------|-------|-----------|-------:|----------|
| `2a69cccc668c` | FP_contamination_corning_cluster | earnings_chip(0.168) | 0.0 | CIEN Targets 30% Revenue Growth Through 2029 — Peers COHR, GLW And AAOI Rally After Guidan |
| `2d83a5133335` | FP_contamination_critic_wrong_theme | ai_compute(0.162) | 0.0 | Corning Rides on Expanding Partner Base: Will it Boost Prospects? |

### Label rationale

- **FP_contamination_corning_cluster:** GLW/Corning peer cluster in optical/earnings guidance wrap (CIEN/COHR/GLW/AAOI); `expected_action=no_hard_skip`.
- **FP_contamination_critic_wrong_theme:** Corning partner-base headline with critic top=`ai_compute` without genuine AI-compute primary; `expected_action=no_hard_skip`.
- Full `theme_probs` / `relevance_by_symbol` attached from `rudra_dev.bot_decisions` (`source=bot_decisions`); **never fabricated**.
- Existing seed FP/TP rows and prior `source=bot_decisions` nests **preserved** (integrity assert passed).

## FP label mix (after)

- `FP_contamination_corning_cluster`: 15
- `FP_contamination_critic_wrong_theme`: 3
- `FP_contamination_keyword_risk`: 1
- `FP_contamination_summary`: 1

## Paths

- Seed (updated): `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json`
- Backup: `/workspace/briefs/golden_sets/2026-09-14-theme-shock-seed.json.bak-pre-soak-2026-09-17_1628`
- Mine: `/workspace/briefs/golden_sets/_soak_mine_2026-09-17.json`
- This brief: `/workspace/briefs/2026-09-17-book-a-fp-soak-delta.md`

## Notes

- FP count hit GO bar (≥20).
- Never flip `THEME_HARD_SKIP_REQUIRE_GPU`; Stage A floors unchanged.
- LIVE stays NO-GO; this is golden-set growth only.
