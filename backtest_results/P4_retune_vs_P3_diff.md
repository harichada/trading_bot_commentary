# P4 barrier-retune vs P3 vs P4 baseline — calibration experiment

Generated: 2026-04-27.

**Experiment:** widen P4 triple-barrier parameters; rerun the existing
P3 nested-CV regime-routing pipeline against the retuned primary.
Hypothesis: gross edge (~0.20R) is real but eaten by 0.32R cost-in-R.
Wider barriers should halve cost-in-R while preserving gross edge,
flipping at least one regime from PF ~0.83 to PF ≥ 1.3.

**Files:**
- `ml_model_v2_retune.pkl` — retuned bundle (released `ml_model_v2.pkl` byte-identical and untouched).
- `backtest_results/P4_retune_training_report.json` — primary CV training report.
- `backtest_results/P4_retune_after.json` — P3 nested-CV regime-routing report.
- `backtest_results/P3_after.json` — comparison baseline.
- `backtest_results/P4_after.json` — original P4 baseline.

**Param deltas (the only changes):**

| Param | Released | Retune | Delta |
| --- | ---: | ---: | --- |
| `pt_mult` | 2.0 | 2.0 | unchanged |
| `sl_mult` | 1.0 | **2.0** | widened 2× |
| `cusum_h` | 0.01 | **0.015** | scaled 1.5× (matches "2.0 × ewma_vol → 3.0 × ewma_vol") |
| `vertical_mult_duration` | 2.0 | **3.0** | widened 1.5× |
| `min_ret_atr_mult` | 1.0 | 1.0 | unchanged |

`primary_bundle_sha256.unchanged = True` for `ml_model_v2.pkl` (mirrors P1/P3 contract — released bundle never modified).

---

## Headline metrics — 4-way comparison

| Metric | P4 baseline | P3 routing | P4-retune routing | Δ vs P3 |
| --- | ---: | ---: | ---: | ---: |
| Aggregate PF (median per-fold) | 0.738 | 0.726 | **0.838** | **+0.112** |
| Aggregate expectancy (R/trade, net) | −0.196 | −0.219 | **−0.083** | **+0.136R** |
| Total trade count | ~1750 | 1502 | 818 | −684 |
| Median sample cost-in-R | ~0.32 | ~0.32 | **0.16** | **−0.16R (halved exactly)** |
| Median trade horizon (bars) | ~6 | ~6 | **10** | +4 |
| Total events (CUSUM-filtered) | 2123 | 2123 | 1125 | −998 |
| % bars classified chop | n/a | 10.2% | 10.2% | unchanged |

**Hypothesis verdict (mechanical):** the cost-halving prediction is **correct exactly** — wider 2× sl barrier doubles 1R, so the same dollar cost becomes half the R. Expectancy improvement of +0.136R/trade is consistent with halving the cost-drag on roughly the same gross edge.

---

## Acceptance gates A–D

| # | Gate | Threshold | Retune result | Verdict |
| ---: | --- | --- | --- | :---: |
| **A** | At least one regime: PF ≥ 1.3 AND exp > 0 AND n ≥ 100 | hard | best is `trend_dn_high_vol` PF=0.844, exp=−0.081R, n=137 | **✗ FAIL** |
| B | Per-fold AR(1) within surviving regimes: median<0.10, max<0.20 | hard | moot — A failed | **moot** |
| C | Chop trade count == 0 across folds | hard | 0 trades | **✓ PASS** |
| D | Aggregate ≥ P4 baseline (PF ≥ 0.738, exp ≥ −0.196R) | sanity | PF=0.838, exp=−0.083R | **✓ PASS** (was FAIL on P3) |

**Headline verdict:** D flipped from FAIL → PASS, A still FAIL. Per spec: STOP, do not relax, do not iterate further on barrier widths without explicit approval.

---

## Per-regime breakdown — retune

| Regime | n_trades | PF | Expectancy | Win-rate | AR1 med | AR1 max | Median gross R | Δ exp vs P3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trend_up_low_vol | 95 | 0.790 | −0.117 | 0.522 | None | None | −1.000 | +0.003 |
| trend_up_high_vol | 126 | 0.498 | −0.302 | 0.357 | +0.356 | 0.356 | −1.000 | **−0.107** |
| trend_dn_low_vol | 115 | 0.682 | −0.190 | 0.485 | −0.177 | 0.274 | +1.000 | +0.085 |
| trend_dn_high_vol | 137 | 0.844 | −0.081 | 0.520 | +0.091 | 0.091 | +1.000 | **+0.283** |
| range_tight | 156 | 0.543 | −0.303 | 0.429 | +0.126 | 0.298 | −1.000 | −0.100 |
| range_wide | 189 | 0.754 | −0.137 | 0.500 | −0.064 | 0.191 | +0.013 | +0.130 |
| **chop** | **0** | — | — | — | — | — | — | — |

**The leadership regime flipped:**
- P3 best: `trend_up_low_vol` (PF 0.827, wr 56%).
- Retune best: **`trend_dn_high_vol`** (PF 0.844, wr 52%) — same PF, different regime.

The asymmetric directional bias of the released primary survived widening but redistributed across vol cohorts. `trend_up_high_vol` regressed catastrophically (wr 47.5% → 35.7%), while `trend_dn_high_vol` improved markedly (wr 48.9% → 52%). The wider stops + longer vertical may favor capturing developing-trend pullbacks over breakout continuations on this universe.

---

## Diagnostic — cost-in-R per trade

| Metric | P4 baseline | Retune | Δ |
|---|---:|---:|---:|
| Median sample cost-in-R | 0.32 | **0.160** | **−0.160 (exactly halved)** |
| Mean sample cost-in-R | 0.32 | 0.160 | −0.160 |

The mechanical prediction held: doubling the barrier doubled 1R, so the dollar cost / 1R ratio halved. This was the most testable claim in the experiment design, and it matches to three decimals.

---

## Honest reading

1. **Cost-in-R halved exactly as predicted (0.32R → 0.16R).** The mechanical structure of triple-barrier labeling under cost-aware sample weighting is correct; the experiment was a clean test of that math. The +0.136R/trade aggregate-expectancy improvement is the expected consequence.

2. **Gate D flipped from FAIL → PASS** with margin: aggregate PF 0.838 vs P4 baseline 0.738 (+0.100), aggregate exp −0.083R vs −0.196R (+0.113R). Regime routing is no longer destroying value vs the unrouted P4 baseline — it's now adding value.

3. **Gate A still fails by a wide margin.** The best regime is at PF 0.844, ~half the gate's PF≥1.3 threshold. The remaining gap is roughly 0.46 PF or ~0.20R expectancy at the per-regime level. Halving the cost stack (0.32→0.16) gave +0.136R aggregate; we'd need another +0.08R/trade gross edge in at least one regime to reach gate A. There's no cost-side knob left at this barrier width that wouldn't be a clear over-fit.

4. **The leadership regime flipped** (trend_up_low_vol → trend_dn_high_vol). Same primary, same features, just wider barriers — different regime wins. This says the primary's edge is shape-dependent, not direction-dependent: it captures *both* directional moves but with horizon characteristics that depend on barrier geometry. Useful information for P5/P6/P8 feature design.

5. **Win rates redistributed asymmetrically.** trend_up_high_vol crashed from 47.5% → 35.7% (worst-case loss; the wider barriers gave high-vol breakouts time to mean-revert against the long signal). trend_dn_high_vol improved 48.9% → 52% (wider stops let drawdowns play out and reward bearish signals). This is a real signal the primary's residual structure is regime-dependent — exactly the shape that suggested per-regime primaries (out of scope for this experiment).

6. **AR(1) is high in some retune regimes** (trend_up_high_vol AR1 0.356, trend_dn_low_vol AR1_max 0.274) — would have failed gate B for those regimes, but moot since A failed. With ~125–190 trades per regime, AR(1) sample noise is real; not necessarily a label-IID violation.

---

## Decision branch (yours, per spec)

Quoting the original retune spec: *"Gate A fails: STOP. Do not relax. Do not iterate further on barrier widths without my approval. Report honest numbers. We will pivot to Path A (P5/P6/P7/P8 feature work)."*

The retune experiment was the cheapest possible test of "does halving cost-in-R unlock per-regime edge?". The mechanical prediction (cost halves) was correct. The economic prediction (at least one regime crosses PF≥1.3) was wrong by a factor of ~1.5×. The remaining gap is on the gross-edge side, which barriers cannot fix.

**Path A pivot is the indicated next move:** add features (P5 cross-sectional momentum, P6 options skew, P8 microstructure) so the primary can extract more gross edge per signal. The structural finding here — that win-rate redistribution is regime-dependent — argues for features that explicitly encode the regime so the primary can adapt rather than depending on a downstream router.

**Do NOT merge `feat/P4-barrier-retune-experiment`** — gate A failed, retain the experiment as a documented branch and an artefact-only contribution (`ml_model_v2_retune.pkl` exists alongside `ml_model_v2.pkl` for any future re-evaluation, but is not promoted).
