# P3 4-Way Diff — P4 baseline | P1 nested (failed) | P3 aggregate | P3 best regime

Generated: 2026-04-27.

**Files:**
- `backtest_results/P4_after.json` — P4 baseline (PF 0.738, exp −0.196R, ungated).
- `backtest_results/P1_nested_after.json` — P1 nested-CV failure record (AFML §7.4).
- `backtest_results/P3_after.json` — this run.

**Primary bundle SHA256 verification (P3 nested run):**
- before = (recorded in P3_after.json)
- after  = (recorded in P3_after.json)
- unchanged = **True** → `ml_model_v2.pkl` byte-identical pre/post run; the released primary was not modified. Mirrors P1's contract.

**Run config:** 10 symbols × 60 days × 5-min bars × 5 purged-CV folds × 5 sequential-bootstrap iterations × cusum_h=0.005 × decision_threshold=0.55 × stability_n=5.

---

## Headline metrics

| Metric | P4 baseline (no meta) | P1 nested (failed) | P3 aggregate | P3 best regime (trend_up_low_vol) |
| --- | ---: | ---: | ---: | ---: |
| Profit factor (median per-fold) | 0.738 | 0.601 | **0.726** | **0.827** |
| Expectancy (R/trade, net) | −0.196 | −0.241 | **−0.219** | **−0.120** |
| Win rate | n/a | 52.1% | n/a (mixed) | **56.2%** |
| Trades count | ~1750 (all events) | 451 OOS-meta | 1502 | 171 |
| Take rate | ~0.70 | 0.047 | 0.71 | 0.40 |
| Per-fold AR(1) median | 0.092 | 0.076 | n/a (computed per regime) | 0.183 |
| Per-fold AR(1) max | 0.168 | 0.150 | n/a | 0.240 |

---

## Acceptance gates A–D

| # | Gate | Threshold | P3 result | Verdict |
| ---: | --- | --- | --- | :---: |
| **A** | At least one regime: PF ≥ 1.3 AND expectancy > 0 AND n ≥ 100 | hard | best regime PF=0.827 / exp=−0.120R / n=171 | **✗ FAIL** |
| B | Per-fold AR(1) within surviving regimes: median<0.10, max<0.20 | hard | best regime AR1_med=0.183, AR1_max=0.240 (would fail B) | **✗ would-fail** (moot — A failed) |
| C | Chop regime trade count == 0 across all folds | hard | 0 trades | **✓ PASS** |
| D | Aggregate ≥ P4 baseline (PF ≥ 0.738, exp ≥ −0.196R) | sanity | PF=0.726, exp=−0.219R | **✗ FAIL (marginal)** |

**Headline verdict: A fails, C passes, D fails marginally. Process spec: STOP, do not relax, do not move on.**

---

## Per-regime breakdown (full P3 run)

| Regime | n_trades | PF (median) | Expectancy R | Win-rate | AR(1) med | AR(1) max | % bars | Cost-drag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| trend_up_low_vol | 171 | 0.827 | −0.120 | 0.562 | +0.183 | 0.240 | 13.1% | n/a (gross ≤ 0) |
| trend_up_high_vol | 236 | 0.758 | −0.195 | 0.475 | −0.006 | 0.151 | 9.7% | n/a |
| trend_dn_low_vol | 177 | 0.645 | −0.275 | 0.512 | −0.065 | 0.118 | 10.3% | n/a |
| trend_dn_high_vol | 260 | 0.579 | −0.364 | 0.489 | +0.020 | 0.425 | 10.0% | n/a |
| range_tight | 282 | 0.768 | −0.203 | 0.500 | −0.006 | 0.249 | 26.5% | n/a |
| range_wide | 376 | 0.655 | −0.267 | 0.513 | −0.025 | 0.132 | 20.1% | n/a |
| **chop** | **0** | — | — | — | — | — | 10.2% | — |

`%bars` is `pct_committed_per_regime` from `regime_distribution` after stability gate. Sums to 100%. None dominates degenerately, so the gate is operating as designed (not eating signal).

`regime_transition_count = 4025` across 82,248 bars × 10 symbols (≈ 400 transitions per symbol over 60 days). Healthy turnover; not stuck.

---

## Honest reading

1. **Best regime is `trend_up_low_vol`**: PF 0.827, win-rate 56.2% on 171 trades. The win-rate alone confirms the routing hypothesis — directional bias in a steady uptrend produces a measurable >50% hit rate, which is *substantially* above the worst regime (`trend_dn_high_vol` at 48.9%). So the primary does have asymmetric structure: it predicts up-moves in low-vol uptrends better than down-moves in high-vol downtrends.

2. **But the edge does not survive cost drag**. Even the best regime loses 0.120R per trade after the same cost model that beat P4 (224% cost-drag in P4's full run). The P1 attempt confirmed this from a different angle (PF 0.601 OOS); P3 confirms it again with regime conditioning held constant. The primary's gross edge is too thin to absorb the realized retail cost stack at this universe / horizon.

3. **No regime hits gate A**:
   - PF ≥ 1.3: max observed is 0.827 (`trend_up_low_vol`).
   - Expectancy > 0: best is −0.120R.
   - n ≥ 100: ALL non-chop regimes pass this floor (171–376 trades each), so the n-floor is not the binding constraint.

4. **Gate D fails marginally** (PF 0.726 vs 0.738 baseline; exp −0.219R vs −0.196R). The marginal degradation is consistent with a few percent of P4's "informational" trades being suppressed in chop without compensating gain elsewhere.

5. **Per-fold AR(1) — caveat** within the regime-conditional view: `trend_up_low_vol` AR1_max 0.240 and `trend_dn_high_vol` AR1_max 0.425 indicate the residual stream in those regime sub-samples is more autocorrelated than the unconditional P4 stream. Two innocent explanations (and one not-so-innocent):
   - Regime-conditioned subsamples are smaller per fold → more sample-noise in AR(1).
   - Stability gate keeps the bot in the same regime for stretches → consecutive trades share state.
   - **Or** the model's residual structure is regime-dependent (a real signal that a regime-specific primary, not a regime-routed shared primary, is needed). Cannot disentangle from this run alone.

6. **Chop gate worked exactly as designed**: 0 trades, 10.2% of bars excluded, 4025 transitions tracked. No degeneracy.

---

## What this confirms

- **The aggregate-edge-masks-regime-edge hypothesis is partially confirmed** (trend_up_low_vol does have visibly better win-rate than chop/dntrend) **but the magnitude is insufficient to clear cost drag**.
- The released `ml_model_v2.pkl` is byte-identical pre and post run (SHA256 verified by the runner). The harness refit primaries are evaluation artefacts only, never persisted. Mirrors P1.
- AR(1) per-fold gate would not have rescued this run: even if we narrow to the best regime, AR1_max 0.240 exceeds the 0.20 P4 ceiling, and the underlying expectancy is still negative.

## Decision branch (yours, per the original spec)

Quoting the spec: *"If gate A fails — no regime shows edge — STOP. Do not relax, do not move on. Report honestly. Decision is mine: either revisit P4 with new features (P5/P6/P7/P8 path) or accept the bot has no edge on this universe at this configuration."*

Two paths:

**(a) Revisit P4 with new features (P5/P6/P7/P8 path).** The cheapest test of "does upstream edge exist in trending regimes?" is to add features that *encode* the regime explicitly so the primary itself can pivot, rather than routing a shared primary. Specifically:
- P5 cross-sectional momentum/mean-reversion features (rank-based): may add the missing alpha component.
- P6 options skew / put-call ratio (P3's downtrend asymmetry suggests put-flow info would help).
- P8 microstructure features (Amihud illiquidity, VPIN): the high-vol-trend buckets (`trend_*_high_vol`) had the worst metrics — micro features may explain the negative edge there.
- P7 Claude-driven event extractor: useful only if P5/P6/P8 don't close the gap.

**(b) Accept that the bot has no edge on this universe at this configuration.** The cost stack is the binding constraint; either tighten it (smaller participation, larger basket to amortize impact, longer horizons that dilute fee_bps relative to expected R), or change the universe (e.g., a higher-vol asset class where the primary's asymmetric long bias has more headroom over costs).

**Do NOT relax P3 gates.** They performed exactly as designed: surfaced that regime conditioning alone cannot create edge that does not exist upstream.
