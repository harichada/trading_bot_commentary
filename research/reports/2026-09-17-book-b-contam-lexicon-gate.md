# Gate Review: Book B Contam Lexicon (PR #93)

**Date:** 2026-09-17  
**PR:** https://github.com/harichada/trading_bot_commentary/pull/93  
**RCA:** `/workspace/briefs/2026-09-17-book-b-contam-rca.md`  
**Verdict:** **FULL GO**

---

## Verification Summary

| # | Criterion | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `GPU_CRITIC_CONTAM_LEXICON_ENABLE` default False | **PASS** | `core/config.py:3434` returns `False` default |
| 2 | Lexicon bumps `contamination_risk≥0.65` on Corning/GLW/motor-oil/oil-crisis/crude wraps (headline\|summary) unless AI headline trigger | **PASS** | `gpu_news_critic.py:450` `max(current_risk, 0.65)`, checks headline+summary text, AI_COMPUTE_TRIGGERS exception |
| 3 | `FakeCritic.CONTAMINATION_TRIGGERS` mirrored | **PASS** | Lines 469-485 include all RCA tokens; same AI headline exception logic |
| 4 | Tests cover FP eids f59b23116fcb, 9579cf0d1b4f, 2d83a5133335 | **PASS** | `TestBookBContamLexicon` 8/8 tests pass, all 3 eids covered |
| 5 | THEME_HARD_SKIP_REQUIRE_GPU / LIVE / ENABLE_MEAN_REV* untouched | **PASS** | No modifications in diff |
| 6 | Modular safe off-path | **PASS** | Default False, early return when disabled |
| — | No LIVE flips in PR | **PASS** | Verified: no LIVE flag modifications |

---

## Detailed Evidence

### 1. GPU_CRITIC_CONTAM_LEXICON_ENABLE default False

```python
# core/config.py:3430-3434
env_val = os.getenv("GPU_CRITIC_CONTAM_LEXICON_ENABLE")
if env_val is not None:
    return env_val.lower() in ("1", "true", "yes", "on")
return bool(self.manager.get('trading.gpu_critic_contam_lexicon_enable', False))
```

### 2. Lexicon Bump Logic

```python
# analysis/gpu_news_critic.py:441-450
has_lexicon_hit = any(token.lower() in text_lower for token in lexicon)
if not has_lexicon_hit:
    return current_risk

headline_has_ai_trigger = any(
    t in headline_lower for t in self.AI_COMPUTE_TRIGGERS
)
if headline_has_ai_trigger:
    return current_risk

return max(current_risk, 0.65)
```

- Matches on `headline_lower + summary_lower` (text only, NOT symbols_seen)
- AI_COMPUTE_TRIGGERS: anthropic, amodei, openai, frontier ai, ai slowdown, gpu demand, data center, data-center, datacenter, ai compute, ai infrastructure
- Bumps to `max(current_risk, 0.65)` → triggers `WOULD_SUPPRESS_HARD_SKIP` via existing ≥0.6 rule

### 3. FakeCritic Mirror

```python
# analysis/gpu_news_critic.py:468-485
CONTAMINATION_TRIGGERS = [
    "crude oil", "petroleum", "opec", "oil prices", "wti crude", "brent crude",
    "corning", "glw", "nyse:glw", "motor oil", "oil crisis", "wti", "brent",
    "cohr", "cien", "aaoi",
]
```

FakeCritic applies same AI headline exception:
```python
is_contaminated = has_contam_trigger and not headline_has_ai_trigger
```

### 4. Test Coverage

All 8 tests pass:

```
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_corning_plunged_fp_suppressed_when_lexicon_enabled PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_oil_crisis_motor_oil_fp_suppressed_when_lexicon_enabled PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_corning_partner_base_fp_suppressed_when_lexicon_enabled PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_anthropic_ai_tp_not_suppressed_when_lexicon_enabled PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_glw_with_ai_headline_trigger_not_suppressed PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_fake_critic_always_catches_lexicon_for_ci_honesty PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_full_critic_corning_fp_when_enabled PASSED
tests/test_gpu_news_critic.py::TestBookBContamLexicon::test_full_critic_anthropic_tp_preserved_when_enabled PASSED
```

FP eids documented in test docstrings:
- `f59b23116fcb`: "Why Corning Plunged Today"
- `9579cf0d1b4f`: "The Oil Crisis Has Reached Costco's Motor Oil Aisle"
- `2d83a5133335`: "Corning Rides on Expanding Partner Base: Will it Boost Prospects?"

### 5. Locks Unchanged

Verified no modifications to:
- `THEME_HARD_SKIP_REQUIRE_GPU` (stays False)
- `LIVE*` flags
- `ENABLE_MEAN_REV*` flags

CHANGELOG explicitly states:
> - `THEME_HARD_SKIP_REQUIRE_GPU=False` (not flipped)
> - No LIVE knobs changed

### 6. Modular Safe Off-Path

- New config property `GPU_CRITIC_CONTAM_LEXICON_ENABLE` defaults to `False`
- `_apply_contam_lexicon()` returns early when disabled (line 430-431)
- New lexicon list `GPU_CRITIC_CONTAM_LEXICON` is configurable via env or YAML
- No changes to existing code paths when feature is disabled

---

## Verdict

**FULL GO** — PR #93 meets all RCA criteria:

1. Safe default off-path (lexicon disabled by default)
2. Correct contamination bump logic (≥0.65) with AI headline exception
3. FakeCritic mirrored for CI honesty
4. All 3 FP eids tested with explicit assertions
5. No LIVE/THEME_HARD_SKIP_REQUIRE_GPU/MEAN_REV flag changes
6. Modular implementation allows soak before production enablement

PR status: **MERGED** (2026-09-17T20:44:51Z)

---

## Next Steps (from RCA)

1. Enable lexicon in shadow-only on KiddoKingdom soak
2. Re-score Book B scorecard with `GPU_CRITIC_CONTAM_LEXICON_ENABLE=1`
3. Target contam_fp ≤5% on current 20 FPs
4. Continue n-labeled growth toward ≥80 / ≥10 sessions
