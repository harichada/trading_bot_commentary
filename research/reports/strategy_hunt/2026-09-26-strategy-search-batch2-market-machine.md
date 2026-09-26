# Strategy hunt report

Computed: 2026-09-26T12:33:40. Rows: 3970. Symbols: 531. Holdout: the final 6 months were withheld from this search (re-run with --holdout-months 0 for the confirmation test). Grid pins: entry_style=market (one arm of an A/B: every family with that axis searched only this value).


## Verdicts by family

| family | side | PASS | WATCH | FAIL | coldstart WATCH |
|---|---|---|---|---|---|
| low_vol_trend | long | 0 | 0 | 238 | 0 |
| mom_12_1 | long | 0 | 0 | 242 | 0 |
| mom_12_1 | short | 0 | 0 | 199 | 0 |
| mr_20d | long | 0 | 0 | 467 | 0 |
| mr_20d | short | 0 | 0 | 495 | 0 |
| mr_short_term | long | 0 | 0 | 495 | 0 |
| mr_short_term | short | 0 | 0 | 507 | 0 |
| overnight_hold | long | 0 | 31 | 470 | 0 |
| overnight_hold | short | 0 | 27 | 477 | 0 |
| rs_rotation | long | 0 | 0 | 322 | 0 |

## Top lanes by score

| symbol | family | side | verdict | n | exp_r | exp_r_lb | t_stat | pf | wr | max_dd_r | stability | decay | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CVNA | mom_12_1 | long | FAIL | 18 | 1.760 | 0.787 | 2.749 | 7.58 | 0.71 | 1.7 | 1.00 | 1.11 | 1.760 |
| VRT | mom_12_1 | long | FAIL | 20 | 0.660 | 0.238 | 2.431 | 4.38 | 0.75 | 2.6 | 1.00 | 1.02 | 0.660 |
| MSTR | rs_rotation | long | FAIL | 23 | 0.687 | -0.004 | 1.683 | 2.91 | 0.57 | 4.1 | 0.80 | 1.95 | 0.614 |
| QQQ | low_vol_trend | long | FAIL | 24 | 0.681 | 0.229 | 2.492 | 3.05 | 0.67 | 3.1 | 0.80 | 1.32 | 0.609 |
| CVNA | rs_rotation | long | FAIL | 23 | 0.507 | -0.042 | 1.531 | 2.45 | 0.57 | 2.2 | 1.00 | 1.73 | 0.507 |
| UBER | mom_12_1 | long | FAIL | 13 | 0.451 | 0.019 | 1.660 | 2.74 | 0.62 | 2.0 | 1.00 | 0.82 | 0.451 |
| RKLB | mom_12_1 | long | FAIL | 14 | 0.390 | -0.065 | 1.383 | 2.74 | 0.57 | 1.3 | 0.75 | 1.34 | 0.338 |
| PLTR | mom_12_1 | long | FAIL | 23 | 0.409 | 0.172 | 2.640 | 4.84 | 0.76 | 1.8 | 0.67 | 1.73 | 0.334 |
| HOOD | mom_12_1 | long | FAIL | 21 | 0.350 | -0.086 | 1.386 | 2.12 | 0.60 | 3.0 | 0.71 | 0.68 | 0.296 |
| NVDA | mom_12_1 | long | FAIL | 25 | 0.344 | 0.051 | 1.413 | 2.03 | 0.60 | 3.0 | 0.57 | 1.87 | 0.260 |
| ROKU | mom_12_1 | short | FAIL | 15 | 0.268 | -0.062 | 1.344 | 2.64 | 0.67 | 2.0 | 0.80 | 0.51 | 0.239 |
| PLTR | rs_rotation | long | FAIL | 16 | 0.259 | -0.158 | 0.951 | 1.79 | 0.62 | 2.8 | 0.75 | 0.96 | 0.224 |
| AFRM | mr_short_term | short | FAIL | 14 | 0.219 | 0.049 | 1.958 | 4.53 | 0.83 | 0.6 | 1.00 | 1.13 | 0.219 |
| SMCI | mom_12_1 | long | FAIL | 24 | 0.255 | -0.123 | 1.102 | 1.90 | 0.57 | 2.2 | 0.71 | 1.99 | 0.216 |
| TDOC | mom_12_1 | short | FAIL | 19 | 0.275 | 0.032 | 1.989 | 2.92 | 0.74 | 1.1 | 0.60 | 0.55 | 0.213 |
| WOLF | mom_12_1 | short | FAIL | 21 | 0.262 | -0.052 | 1.191 | 2.20 | 0.53 | 1.5 | 0.60 | 0.81 | 0.203 |
| APP | rs_rotation | long | FAIL | 29 | 0.228 | -0.132 | 1.119 | 1.74 | 0.61 | 4.4 | 0.75 | 0.79 | 0.198 |
| JPM | low_vol_trend | long | FAIL | 24 | 0.262 | -0.146 | 0.968 | 1.61 | 0.58 | 3.4 | 0.50 | 0.40 | 0.185 |
| PTON | mom_12_1 | short | FAIL | 22 | 0.225 | -0.024 | 1.455 | 2.20 | 0.64 | 1.4 | 0.67 | 0.53 | 0.184 |
| FSLY | mom_12_1 | short | FAIL | 14 | 0.181 | -0.096 | 1.136 | 2.17 | 0.77 | 2.1 | 1.00 | 2.12 | 0.181 |
| META | mom_12_1 | long | FAIL | 15 | 0.245 | -0.164 | 0.975 | 2.03 | 0.64 | 2.4 | 0.50 | 1.05 | 0.173 |
| INFY | mr_short_term | short | FAIL | 20 | 0.198 | 0.058 | 2.103 | 3.99 | 0.65 | 0.5 | 0.75 | 1.15 | 0.171 |
| XP | mr_short_term | short | FAIL | 17 | 0.188 | -0.039 | 1.203 | 2.31 | 0.69 | 1.3 | 0.75 | 0.76 | 0.163 |
| DVN | mom_12_1 | long | FAIL | 12 | 0.178 | -0.315 | 0.526 | 1.40 | 0.50 | 2.0 | 0.75 | 1.00 | 0.155 |
| PYPL | mr_short_term | short | FAIL | 27 | 0.203 | 0.089 | 2.738 | 4.41 | 0.76 | 0.8 | 0.57 | 0.50 | 0.153 |
| SMCI | mr_short_term | long | FAIL | 28 | 0.166 | 0.028 | 1.783 | 3.46 | 0.65 | 0.6 | 0.75 | 0.36 | 0.143 |
| ZTO | overnight_hold | short | FAIL | 43 | 0.168 | 0.030 | 1.448 | 3.85 | 0.64 | 0.6 | 0.67 | 0.28 | 0.137 |
| COIN | mom_12_1 | long | FAIL | 15 | 0.274 | -0.179 | 0.815 | 1.90 | 0.53 | 2.7 | 0.25 | 0.85 | 0.137 |
| TSM | overnight_hold | long | FAIL | 37 | 0.122 | 0.028 | 1.698 | 3.75 | 0.67 | 0.5 | 1.00 | 1.31 | 0.122 |
| FXI | overnight_hold | long | WATCH | 73 | 0.130 | 0.056 | 2.567 | 2.35 | 0.58 | 1.9 | 0.88 | 1.47 | 0.121 |

## Pooled out-of-sample by family

| family | side | n | exp_r | pf | wr |
|---|---|---|---|---|---|
| low_vol_trend | long | 1582 | -0.040 | 0.92 | 0.41 |
| mom_12_1 | long | 1420 | 0.130 | 1.35 | 0.49 |
| mom_12_1 | short | 1498 | -0.111 | 0.76 | 0.50 |
| mr_20d | long | 957 | 0.112 | 1.29 | 0.51 |
| mr_20d | short | 1114 | -0.091 | 0.83 | 0.42 |
| mr_short_term | long | 4377 | -0.003 | 0.98 | 0.50 |
| mr_short_term | short | 5637 | 0.015 | 1.08 | 0.52 |
| overnight_hold | long | 25414 | -0.010 | 0.87 | 0.47 |
| overnight_hold | short | 29904 | -0.013 | 0.81 | 0.45 |
| rs_rotation | long | 1686 | 0.064 | 1.18 | 0.48 |

## Per-regime expectancy by family

| family | side | mixed | risk_off | risk_on |
|---|---|---|---|---|
| low_vol_trend | long | -0.076 (n=1487) | 0.457 (n=74) | 0.749 (n=21) |
| mom_12_1 | long | 0.060 (n=1302) | 0.656 (n=84) | 1.516 (n=34) |
| mom_12_1 | short | -0.091 (n=1366) | -0.505 (n=84) | -0.008 (n=48) |
| mr_20d | long | 0.055 (n=641) | 0.237 (n=302) | 0.038 (n=14) |
| mr_20d | short | -0.102 (n=1051) | 0.397 (n=50) | -1.037 (n=13) |
| mr_short_term | long | -0.005 (n=3490) | 0.001 (n=812) | 0.049 (n=75) |
| mr_short_term | short | 0.009 (n=5181) | 0.029 (n=132) | 0.107 (n=324) |
| overnight_hold | long | -0.006 (n=14807) | -0.032 (n=6485) | 0.012 (n=4122) |
| overnight_hold | short | -0.020 (n=19180) | 0.002 (n=735) | 0.000 (n=9989) |
| rs_rotation | long | 0.036 (n=1470) | 0.273 (n=151) | 0.208 (n=65) |

## Pooled rows vs baseline

| family | side | regime | baseline | n | exp_R | baseline exp_R | baseline n | replayed | excess | t_session | recovery | verdict | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | long | risk_on | selection | 34 | 1.516 | 0.230 | 30 | 0.882 | 1.286 | 1.82 | 7.47 | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session, baseline_underreplayed, baseline_stale |
| mom_12_1 | long | risk_off | selection | 84 | 0.656 | 0.122 | 75 | 0.893 | 0.534 | 1.08 | 7.65 | FAIL | min_n, min_sessions, min_t_stat_session, baseline_underreplayed, baseline_stale |
| mr_20d | short | risk_off | selection | 50 | 0.397 | 0.044 | 42 | 0.840 | 0.354 | 0.70 | 3.16 | FAIL | min_n, min_sessions, min_t_stat, min_t_stat_session, baseline_underreplayed, baseline_stale |
| mom_12_1 | short | risk_on | selection | 48 | -0.008 | -0.162 | 41 | 0.854 | 0.153 | -0.20 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_underreplayed, baseline_stale |
| mr_short_term | short | risk_off | selection | 132 | 0.029 | -0.120 | 117 | 0.886 | 0.149 | 0.88 | 0.81 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_underreplayed, baseline_stale |
| mr_short_term | long | risk_on | selection | 75 | 0.049 | -0.085 | 69 | 0.920 | 0.134 | 0.75 | 0.64 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_stale |
| rs_rotation | long | risk_off | selection | 151 | 0.273 | 0.140 | 143 | 0.947 | 0.133 | 1.05 | 3.40 | FAIL | min_n, min_sessions, min_t_stat, min_t_stat_session, baseline_stale |
| mr_20d | long | risk_off | selection | 302 | 0.237 | 0.108 | 270 | 0.894 | 0.129 | 1.80 | 2.05 | WATCH | min_n, min_recovery, min_stability, min_t_stat_session, baseline_underreplayed, baseline_stale |
| rs_rotation | long | risk_on | selection | 65 | 0.208 | 0.092 | 62 | 0.954 | 0.116 | 0.59 | 1.55 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_stale |
| mom_12_1 | long | mixed | selection | 1302 | 0.060 | 0.017 | 1222 | 0.939 | 0.043 | 1.37 | 0.67 | WATCH | min_pf, min_recovery, min_t_stat, min_t_stat_session, min_excess_r, baseline_stale |
| mr_20d | long | mixed | selection | 641 | 0.055 | 0.013 | 612 | 0.955 | 0.042 | -0.66 | 1.08 | WATCH | min_pf, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| mr_short_term | short | risk_on | selection | 324 | 0.107 | 0.068 | 314 | 0.969 | 0.040 | -1.02 | 1.37 | WATCH | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r, baseline_stale |
| mr_short_term | long | risk_off | selection | 812 | 0.001 | -0.038 | 759 | 0.935 | 0.039 | 0.01 | 0.02 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| rs_rotation | long | mixed | selection | 1470 | 0.036 | 0.008 | 1402 | 0.954 | 0.028 | 0.60 | 0.69 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| mr_short_term | short | mixed | selection | 5181 | 0.009 | -0.008 | 4992 | 0.964 | 0.016 | -1.42 | 0.30 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| overnight_hold | short | mixed | selection | 19180 | -0.020 | -0.030 | 18313 | 0.955 | 0.009 | -4.54 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| overnight_hold | long | risk_on | selection | 4122 | 0.012 | 0.005 | 3945 | 0.957 | 0.007 | -0.16 | 1.33 | WATCH | min_exp_r, min_recovery, min_decay, min_t_stat_session, min_excess_r, baseline_stale |
| low_vol_trend | long | mixed | selection | 1487 | -0.076 | -0.080 | 1413 | 0.950 | 0.004 | 0.23 | 0.00 | FAIL | min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| overnight_hold | short | risk_on | selection | 9989 | 0.000 | -0.003 | 9679 | 0.969 | 0.004 | -1.92 | 0.02 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| overnight_hold | long | mixed | selection | 14807 | -0.006 | -0.008 | 14206 | 0.959 | 0.001 | -0.02 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| overnight_hold | long | risk_off | selection | 6485 | -0.032 | -0.030 | 6281 | 0.969 | -0.002 | -1.46 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| overnight_hold | short | risk_off | selection | 735 | 0.002 | 0.022 | 707 | 0.962 | -0.020 | -1.63 | 0.11 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| mr_short_term | long | mixed | selection | 3490 | -0.005 | 0.034 | 3257 | 0.933 | -0.039 | 0.09 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| mr_20d | short | mixed | selection | 1051 | -0.102 | -0.030 | 976 | 0.929 | -0.073 | -1.49 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| mom_12_1 | short | mixed | selection | 1366 | -0.091 | -0.005 | 1304 | 0.955 | -0.086 | -2.02 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| low_vol_trend | long | risk_off | selection | 74 | 0.457 | 0.584 | 70 | 0.946 | -0.128 | 2.01 | 1.89 | FAIL | min_n, min_sessions, min_recovery, min_t_stat, min_t_stat_session, min_excess_r, baseline_stale |
| mr_20d | long | risk_on | selection | 14 | 0.038 | 0.211 | 11 | 0.786 | -0.173 | 0.09 | 0.07 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed, baseline_stale |
| mom_12_1 | short | risk_off | selection | 84 | -0.505 | -0.143 | 78 | 0.929 | -0.362 | -1.88 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_stale |
| low_vol_trend | long | risk_on | selection | 21 | 0.749 | 1.419 | 21 | 1.000 | -0.670 | 0.16 | 5.28 | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r, baseline_stale |
| mr_20d | short | risk_on | selection | 13 | -1.037 | -0.009 | 11 | 0.846 | -1.029 | -5.09 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed, baseline_stale |

## Search accounting

| family | grid combos × sides | unconditional rows scored | PASS | WATCH |
|---|---|---|---|---|
| low_vol_trend | 2 × 1 = 2 | 238 | 0 | 0 |
| mom_12_1 | 2 × 2 = 4 | 441 | 0 | 0 |
| mr_20d | 12 × 2 = 24 | 962 | 0 | 0 |
| mr_short_term | 6 × 2 = 12 | 1002 | 0 | 0 |
| overnight_hold | 8 × 2 = 16 | 1005 | 0 | 58 |
| rs_rotation | 4 × 1 = 4 | 322 | 0 | 0 |
| **total** | 62 | 3970 | 0 | 58 |

6 families, 62 grid points × sides evaluated, 0 PASS

## Controls

_no control runs found_ (real PASS rows: 0)

## Symbols with no passing lane

531 of 531 symbols have no PASS lane.

AA, AAL, AAPL, ABBV, ABEV, ABNB, ABT, ACHR, ACI, ADBE, ADI, ADM, AEO, AES, AFRM, AG, AGL, AGNC, AI, AIG, ALB, ALLY, AMAT, AMC, AMD, AMDL, AMZN, ANET, APA, APH, APLD, APO, APP, APTV, AR, ARKK, ARM, ARRY, ASAN, ASO, ATAI, AUR, AVGO, AVTR, AXP, B, BA, BABA, BAC, BAX

## Portfolio replay of PASS lanes

- taken: 0
- skipped (capacity): 0
- skipped (symbol already open): 0
- skipped (size too small): 0
- final equity: 43500.00
- total return: 0.00%
- max drawdown: 0.0%
- daily Sharpe: 0.00

### By family

_no trades taken_

### Equity curve (monthly)

_no data_

## Cost calibration

- n: 108
- median slippage fraction: 0.0001
- p75 slippage fraction: 0.0010
- usable: True

### By bucket

| bucket | n | median_frac |
|---|---|---|
| large | 22 | 0.0006 |
| mega | 52 | 0.0000 |
| mid | 32 | 0.0000 |
| small | 2 | 0.0009 |

## Caveats

- Universe is selected by today's data availability (survivorship); results are stated for symbols that exist now.
- Spread and slippage are estimated from bars; see the cost controls for sensitivity.
- Ten families times up to 27 parameter sets across hundreds of symbols will produce spurious PASS rows; the in-root controls estimate how much of a pooled lane's expectancy is the regime's drift -- the path shuffle for intraday timing, the random-peer replay for selection (see "Pooled rows vs baseline").
- The selection control's symbol coverage is a random draw, not a completeness measure, so it is deliberately exempt from the baseline coverage guard; what a dropped trade does to it is measured per group by the `replayed` column and `baseline_underreplayed` instead.
- Live indicators are computed on the in-progress bar; research uses closed bars. New lanes must use closed bars.

