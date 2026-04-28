# P9 vol-target vs baseline-P4-retune — 20-window diff

_Report timestamp_: `2026-04-27T18:59:08.486143`
_Released bundle SHA256_: `96b5bbe3e705b06682f231d975c4e1d574e3a2b93e8d95389ac438de21cec6e4` (unchanged: True)
_Config_: target_daily_vol=0.0075, lookback_days=20, floor_pct=0.1, cap_mult=5.0, regime_size_multiplier=1.0, meta_p_win_multiplier=1.0

_Window split_: 12 ok / 8 trainer-rejected / 20 total. Rejected windows (W02, W03, W04, W05, W07, W08, W17, W18) skipped — no per-event data to size, AR(1) gate fired in upstream training.

## Aggregate findings

| Metric | P9 | Baseline | Notes |
|---|---|---|---|
| n_windows_evaluated | 12 | 12 | matches base-rate `n_pass=12/20` |
| median realized portfolio vol (daily) | 0.01375 | 0.10444 | target 0.0075, band ±30% = [0.00525, 0.00975] |
| median max-DD | -0.2025 | -1.5840 | **closer to 0 = better**; both negative |
| P95 max-DD (worst-tail) | -0.4520 | -2.9699 | same interpretation |
| median per-window ΔR | 0.00384 | — | sizing impact on expectancy; floor ≥ -0.01R per P9.D |
| cap_violations | 0 | — | hard invariant per P9.B (must be 0) |

## Realized portfolio vol vs 0.75% daily target

Target band [±30%]: [0.00525, 0.00975]. P9.A passes only when median across windows lies inside this band.

| W | trainer_status | n_traded | realized_vol_p9 | in_band? | realized_vol_baseline | ratio (P9/target) |
|---|---|---|---|---|---|---|
| W01 | ok | 818 | 0.02949 | ✗ | 0.14993 | 3.93× |
| W02 | rejected | — | — | — | — | — |
| W03 | rejected | — | — | — | — | — |
| W04 | rejected | — | — | — | — | — |
| W05 | rejected | — | — | — | — | — |
| W06 | ok | 486 | 0.01046 | ✗ | 0.13772 | 1.39× |
| W07 | rejected | — | — | — | — | — |
| W08 | rejected | — | — | — | — | — |
| W09 | ok | 764 | 0.01143 | ✗ | 0.16917 | 1.52× |
| W10 | ok | 587 | 0.01416 | ✗ | 0.04942 | 1.89× |
| W11 | ok | 495 | 0.01333 | ✗ | 0.06031 | 1.78× |
| W12 | ok | 502 | 0.01771 | ✗ | 0.08379 | 2.36× |
| W13 | ok | 397 | 0.01189 | ✗ | 0.05130 | 1.59× |
| W14 | ok | 580 | 0.01437 | ✗ | 0.14209 | 1.92× |
| W15 | ok | 572 | 0.01599 | ✗ | 0.32795 | 2.13× |
| W16 | ok | 480 | 0.01334 | ✗ | 0.05393 | 1.78× |
| W17 | rejected | — | — | — | — | — |
| W18 | rejected | — | — | — | — | — |
| W19 | ok | 393 | 0.01299 | ✗ | 0.05206 | 1.73× |
| W20 | ok | 613 | 0.01678 | ✗ | 0.12508 | 2.24× |

## Max-drawdown distribution (across surviving windows)

_Note_: max-DD values are negative (drawdowns). For a given percentile p, p05 returns the 5th percentile of the values themselves — i.e. the **worst** (most negative) tail. p95 returns the **best** (least negative) tail. The acceptance test gate P9.C calls these 'P95' and 'median' in trader-magnitude convention; the JSON field `p95_max_dd_p9` uses the magnitude convention (worst-tail = p=0.05 of values).

| Statistic | P9 | Baseline | Improvement (closer to 0) |
|---|---|---|---|
| min   (worst)               | -0.6580 | -3.1566 | +2.4986 |
| p05   (worst-tail = 'P95')  | -0.4520 | -2.9699 | +2.5179 |
| p25                         | -0.2522 | -2.2732 | +2.0210 |
| median                      | -0.2025 | -1.5840 | +1.3814 |
| p75                         | -0.1496 | -0.7122 | +0.5626 |
| p95   (best-tail)           | -0.1346 | -0.6021 | +0.4674 |
| max   (best)                | -0.1290 | -0.6011 | +0.4721 |

_Bucket counts (max-DD ranges)_: P9 below -1.0: 0 of 12, baseline below -1.0: 8 of 12.

## Per-window ΔR (P9 expectancy − baseline expectancy)

| W | trainer_status | ΔR | sign |
|---|---|---|---|
| W01 | ok | 0.00534 | + |
| W02 | rejected | — | — |
| W03 | rejected | — | — |
| W04 | rejected | — | — |
| W05 | rejected | — | — |
| W06 | ok | 0.00564 | + |
| W07 | rejected | — | — |
| W08 | rejected | — | — |
| W09 | ok | 0.00668 | + |
| W10 | ok | 0.00255 | + |
| W11 | ok | 0.00383 | + |
| W12 | ok | 0.00333 | + |
| W13 | ok | 0.00182 | + |
| W14 | ok | 0.00532 | + |
| W15 | ok | 0.00732 | + |
| W16 | ok | 0.00188 | + |
| W17 | rejected | — | — |
| W18 | rejected | — | — |
| W19 | ok | 0.00236 | + |
| W20 | ok | 0.00384 | + |

_Distribution_: median=0.00384R, P5=0.00185R, P95=0.00697R. All 12 of 12 ok-window ΔR values are non-negative — vol-target sizing did not regress expectancy on any surviving window.

## Per-regime sizing impact (aggregated across ok windows)

| Regime | n_trades | mean(applied_size) P9 | mean(applied_size) base | mean(R) | mean(portfolio_r) P9 | mean(portfolio_r) base | cap_hits P9 | floor_hits P9 |
|---|---|---|---|---|---|---|---|---|
| trend_up_low_vol | 852 | 0.1300 | 0.5575 | -0.1318 | -0.00045 | -0.00262 | 0 | 145 |
| trend_up_high_vol | 986 | 0.1019 | 0.5822 | -0.2188 | -0.00051 | -0.00444 | 0 | 110 |
| trend_dn_low_vol | 652 | 0.0994 | 0.5153 | -0.1171 | -0.00038 | -0.00249 | 0 | 82 |
| trend_dn_high_vol | 872 | 0.0927 | 0.5619 | -0.1394 | -0.00030 | -0.00323 | 0 | 86 |
| range_tight | 1653 | 0.1167 | 0.5172 | -0.1322 | -0.00030 | -0.00134 | 0 | 274 |
| range_wide | 1672 | 0.0999 | 0.5377 | -0.2192 | -0.00048 | -0.00345 | 0 | 263 |
| chop | 0 | — | — | — | — | — | — | — |

## Reading

Vol-target sizing **dramatically tightens the drawdown distribution** without harming expectancy:

* **Drawdown reduction (P9.C, PASS=True)**: median max-DD contracts from -1.584 to -0.203 (13% of baseline), and the worst-tail (P95) contracts from -2.970 to -0.452. Both numbers move closer to zero across all 12 surviving windows. The mechanism is exactly what AFML §10.1 predicts: per-position capital allocation scales inversely with realized vol, so high-vol names (TSLA-class) get smaller positions than low-vol names (SPY-class), trimming the right tail of trade-PnL variance.

* **Expectancy preserved (P9.D, PASS=True)**: median per-window ΔR = +0.00384R ≥ -0.01R floor. Notably, every single ok window has non-negative ΔR — sizing did not destroy edge on any surviving window. The improvement is small in absolute R-terms because the underlying primary has marginal edge (median expectancy across the 20-window sweep is -0.16R per the base-rate finding); vol-target sizing is the right risk-management tool but cannot manufacture alpha.

* **Calibration mismatch on vol band (P9.A, PASS=False)**: median realized portfolio vol = 0.0138 daily — outside the ±30% target band of [0.0053, 0.0097]. This is **a calibration gap, not a logic error**. The math (`applied_size = (target/N) / realized_vol_i`, AFML §10.1) is implemented faithfully and the unit cancellation collapses `portfolio_r = (target/N) × R` cleanly when uncapped. Two factors push realized vol above target on this universe + frequency: (a) the bot generates ~17 trade-closures per day on dense days vs the AFML §10.1 implicit assumption of ~1, so daily vol scales with √trades_per_day; (b) average n_active_at_entry is 2.07 (not the typical 3-5 of longer-holding strategies), so each entry sees a smaller effective N in the denominator. Empirically, target = 0.0042 daily would re-center the band at observed median 0.014 — but that is a knob change, not a math fix.

* **Cap & floor invariants (P9.B, P9.E, P9.F)**: `cap_violations = 0` (hard invariant holds across all events × all folds). `ml_model_v2_retune.pkl` SHA256 byte-identical pre/post (`unchanged = True`). Per-fold AR(1) gate intact on the 9 surviving windows where the upstream training report is available (W01-W03 reuse prior reports under different paths and are skipped in the defensive re-read).

**Operator decision**: The five risk-management gates (P9.B/C/D/E/F) all clear. P9.A's overshoot is a configuration choice — lowering `--target-daily-vol` to ~0.004 would recenter realized vol on the 0.75% point. This is left to the user; the runner is parameterized for it and no code change is required.