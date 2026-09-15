# Late-Entry Gate — Promote Checklist

**Owner:** CoS / Research / Engine  
**Date:** 2026-09-15  
**Refs:** PR #65 (gate impl), `core/config.py` late-entry flags  
**Hari lock:** LIVE disabled — this checklist governs shadow→hard promotion only, not LIVE re-enable.

---

## 1. What the gate does

The late-entry gate prevents chasing extended moves by detecting three conditions:

| Heuristic | Trigger | Default threshold |
|-----------|---------|-------------------|
| **Extension ratio** | `(price - open) / (high - open) >= k` | k = 0.75 |
| **VWAP chase** | Long entry above `VWAP + m*ATR` | m = 1.0 |
| **Bars-since-impulse** | Breakout/impulse occurred ≥ N bars ago | N = 5 |

**Strategies covered:** `day_trade_momentum`, `mean_reversion`, `mean_reversion_short`, `orb_contraction_rvol`

---

## 2. Shadow vs hard mode

| Mode | Behavior | Config |
|------|----------|--------|
| **Shadow** (default) | Logs `LATE_ENTRY_SKIP` / `action=shadow_late_entry_skip` with reason; entry proceeds | `LATE_ENTRY_GATE_SHADOW=1` |
| **Hard** | Blocks entry; logs `LATE_ENTRY_SKIP` / `action=late_entry_skip` | `LATE_ENTRY_GATE_SHADOW=0` |

Master switch: `ENABLE_LATE_ENTRY_GATE=0` disables all late-entry checks.

---

## 3. Metrics to collect (shadow period)

### Required fields (Engine logs)

| Field | Description |
|-------|-------------|
| `shadow_late_entry_skip` | Count of shadow-skipped entries |
| `late_entry_reason` | Which heuristic(s) triggered |
| `would_entry_price` | Entry price if not skipped |
| `would_stop_loss` | Stop if entered |
| `forward_ret_{5,15,60}m` | Forward returns of skipped candidate |

### Derived metrics

| Metric | Formula |
|--------|---------|
| **Skip rate** | `shadow_skips / total_entry_candidates` |
| **False-block rate** | `skips where forward_ret > +0.5R / total_skips` |
| **Would-have-been losers** | `skips where forward_ret < -0.5R / total_skips` |
| **Heuristic precision** | Per-heuristic: what % of skips avoided a losing trade |

---

## 4. Stage A promote criteria (locked floors)

Gate promotes from shadow to hard **only** when all of the following are met:

### Sample size

- [ ] **n ≥ 150** shadow-skip events
- [ ] **≥ 10 sessions** with ≥ 1 skip each

### Performance (on "would-have-entered" book)

- [ ] **PF ≥ 1.30** — skipped candidates lost more than they won
- [ ] **WR ≥ 48%** — skip precision (skips that avoided loss)
- [ ] **Expectancy ≥ +0.05R** — average R saved per skip

### Risk

- [ ] **DD ≤ 6%** — worst drawdown on would-have-entered book
- [ ] **Max losing day ≤ 2R** — no single-day blowup in skipped candidates

### Scorecard source

- [ ] **LIVE Schwab fills only** — no sim/paper; shadow logs must pair with LIVE fill data when applicable

---

## 5. Go / No-Go decision

### GO criteria (all must pass)

1. All Stage A boxes checked above
2. Research sign-off on forward-return analysis
3. Engine confirms logging coverage (no missing fields)
4. No regression in non-skipped entries (control book PF stable)

### NO-GO triggers

- Skip rate > 40% (gate too aggressive — tighten thresholds first)
- False-block rate > 30% (missing winners — investigate heuristic tuning)
- Forward-ret distribution bimodal (gate catches winners and losers equally)

### Promote action

```bash
# After GO decision — promote to hard-skip
LATE_ENTRY_GATE_SHADOW=0
```

---

## 6. Rollback

If hard mode degrades overall performance:

```bash
# Flip back to shadow
LATE_ENTRY_GATE_SHADOW=1
```

Or disable entirely:

```bash
ENABLE_LATE_ENTRY_GATE=0
```

**Post-rollback:** Research reopens analysis with extended shadow period.

---

## 7. Explicit scope boundary

> **Promote checklist ≠ permission to re-enable LIVE**
>
> This checklist governs **shadow → hard-skip** promotion for the late-entry gate.
>
> Re-enabling LIVE entries (`DAY_TRADE_LIVE_ENTRIES_ENABLED`, `MEAN_REV_LIVE_ENTRIES_ENABLED`, etc.) is a **separate Hari decision** requiring its own Stage A validation — see `PROFITABILITY_ROADMAP.md` and the multi-sentiment Stage A gates in `docs/research/2026-09-14-multisenti-stage-a.md`.

---

## References

- **Implementation:** PR #65, `core/config.py` (lines 1864–1940)
- **Stage A floors:** `docs/research/2026-09-14-multisenti-stage-a.md` §4
- **Profitability roadmap:** `docs/PROFITABILITY_ROADMAP.md`
- **Autonomy profiles:** `docs/ARCHITECTURE.md` §5
