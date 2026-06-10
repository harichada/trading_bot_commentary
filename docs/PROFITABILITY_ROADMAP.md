# Profitability Roadmap

*Drafted 2026-06-10, grounded in the walk-forward evidence of
2026-06-09/10 (`research/walkforward_report_2026-06-09.json`).
Companion to `COMPOSITE_VIEW_ARCHITECTURE.md`.*

---

## 1. The evidence this plan stands on

Six non-overlapping 60-day walk-forward windows, Apr 2025 → Apr 2026,
per-window top-30 universes, direction gates active (harness fixed in
`d96f9b5` + `065eeff`):

| Window | Breakout PF | Mean-rev PF | Winner |
|---|---|---|---|
| 1 Apr–Jun 25 | **1.46** | 1.02 | breakout |
| 2 Jun–Aug 25 | **1.34** | 0.70 | breakout |
| 3 Aug–Oct 25 | **1.40** | **1.34** | both |
| 4 Oct–Dec 25 | 0.58 | 0.98 | neither |
| 5 Dec–Feb 26 | 1.05 | 0.96 | flat |
| 6 Feb–Apr 26 | 0.72 | **1.55** | mean-rev |

**Core finding: the strategies are near-anti-correlated across
regimes.** Each one's catastrophic window is the other's good window.
Neither has a standalone edge; the edge — if we can capture it — is in
*allocating between them by regime*. Momentum stays disabled (PF
0.84–0.85, by design since 2026-04-20).

## 2. Root causes of unprofitability

1. **Regime blindness at the allocation level** — the risk-on/off gate
   decides *whether* to trade, never *which strategy* fits the tape.
2. **Long-only book** — no participation in down/chop tapes; pure
   long beta.
3. **No continuous per-strategy/per-regime attribution** — a strategy
   can bleed for weeks in its bad regime before anyone notices.
4. **Exit inefficiency** — WR 37–45% at PF ≈ 1 means winners don't run;
   historical momentum data showed 31–35% of trades timing out.
5. **Cost drag at quarter size** — spread+slippage sinks PF-1.1 edges.
6. **Operator alpha not encoded** — manual trades (regime + news +
   extension reads) outperform the bot; that skill is encodable and is
   the Composite View mandate.

## 3. Architecture law for all additions

Every module ships as a **pluggable, evidence-gated unit**:

- config flag, default OFF
- **shadow mode first** — logs decisions, changes nothing
- append-only NDJSON audit ledger
- a written decision gate ("promotes to live when X")

```
                   ┌─────────────────────────────────────┐
                   │ ALLOCATION LAYER (new)              │
 MarketContext ──▶ │ allocators/regime_allocator.py      │
                   │ per-strategy enable + size weights  │
                   └────────────────┬────────────────────┘
                                    ▼
 Strategies (existing) ─▶ signal ─▶ GATE CHAIN (formalize gates/)
                                    ▼
                          EXIT ENGINE (new, exits/)
                                    ▼
                          ATTRIBUTION (new, attribution/)
```

## 4. Workstreams, by expected dollars per unit of work

| # | Module | Effort | Risk | Decision gate |
|---|---|---|---|---|
| P1 | **Regime allocator** `allocators/regime_allocator.py` | ~1 wk | none (shadow) | regime-allocated walk-forward beats both standalone PFs; then 2-wk live shadow agreement |
| P2 | **SHORT mirrors walk-forward** (breakdown-short exists: `ENABLE_SHORT_MIRRORS`) | ~2 d | none (research) | PF ≥ 1.3 across windows → enters shadow pipeline like #37 |
| P3 | **Attribution ledger** `attribution/` | ~3 d | none | always-on; powers kill-switches + sizing decisions |
| P4 | **Exit engine** `exits/` pluggable policies, A/B in replay | ~1 wk | replay-validated | candidate exit beats ATR bracket out-of-sample |
| P5 | **Composite View Phase 1** (ADR written) | 2–4 wk | read-only | per ADR Article 5 |
| P6 | **Cost model + min-edge filter** | ~2 d | low | expected edge ≥ 2× estimated cost or skip |

**P1 validation comes before P1 code**: re-run the walk-forward with a
simple regime signal (e.g., SPY 20-day efficiency ratio — trending →
breakout, choppy → mean-rev) selecting which strategy may trade. If
the allocated equity line doesn't beat both standalone lines, the
cornerstone hypothesis is wrong and we stop before building.

## 5. Staged definition of "profitable"

1. **Stage 1 (≈4 wks): positive expectancy at quarter size.**
   Bot-only PF ≥ 1.2 over 30+ live trades, allocator shadow agreeing.
2. **Stage 2: scale what's measured.** Size 0.25 → 0.5 → 1.0 per
   strategy, on attribution evidence only.
3. **Stage 3: two-sided book.** SHORT live after soak; bot becomes
   regime-adaptive instead of a leveraged long bet.

Honest ceiling: on a ~$33k account Stage 1 is hundreds/week, not
thousands. These four weeks buy the *right to size up*, not income.

## 6. Standing discipline (unchanged)

- Nothing touches live money without shadow evidence
- No mid-session deploys
- Every veto/skip is logged and answerable
- Strategies are limbs; the regime/context read is the brain
