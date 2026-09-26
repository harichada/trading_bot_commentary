# Strategy hunt report

Computed: 2026-09-26T12:35:57. Rows: 3779. Symbols: 530. Holdout: the final 6 months were withheld from this search (re-run with --holdout-months 0 for the confirmation test). Grid pins: entry_style=limit_open (one arm of an A/B: every family with that axis searched only this value).


## Verdicts by family

| family | side | PASS | WATCH | FAIL | coldstart WATCH |
|---|---|---|---|---|---|
| low_vol_trend | long | 0 | 0 | 238 | 0 |
| mom_12_1 | long | 0 | 0 | 232 | 0 |
| mom_12_1 | short | 0 | 0 | 192 | 0 |
| mr_20d | long | 0 | 0 | 427 | 0 |
| mr_20d | short | 0 | 0 | 458 | 0 |
| mr_short_term | long | 0 | 0 | 469 | 0 |
| mr_short_term | short | 0 | 0 | 492 | 0 |
| overnight_hold | long | 0 | 22 | 463 | 0 |
| overnight_hold | short | 0 | 21 | 465 | 0 |
| rs_rotation | long | 0 | 0 | 300 | 0 |

## Top lanes by score

| symbol | family | side | verdict | n | exp_r | exp_r_lb | t_stat | pf | wr | max_dd_r | stability | decay | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CVNA | mom_12_1 | long | FAIL | 18 | 1.767 | 0.795 | 2.760 | 7.67 | 0.71 | 1.7 | 1.00 | 1.11 | 1.767 |
| QQQ | low_vol_trend | long | FAIL | 24 | 0.681 | 0.229 | 2.492 | 3.05 | 0.67 | 3.1 | 0.80 | 1.32 | 0.609 |
| PLTR | mom_12_1 | long | FAIL | 19 | 0.394 | 0.152 | 2.167 | 4.20 | 0.76 | 1.7 | 0.60 | 1.18 | 0.305 |
| CVNA | rs_rotation | long | FAIL | 21 | 0.301 | -0.195 | 0.925 | 1.80 | 0.53 | 2.2 | 1.00 | 1.64 | 0.301 |
| APP | rs_rotation | long | FAIL | 20 | 0.330 | -0.089 | 1.211 | 2.08 | 0.60 | 2.3 | 0.75 | 0.43 | 0.285 |
| NVDA | mom_12_1 | long | FAIL | 23 | 0.434 | 0.031 | 1.686 | 2.39 | 0.65 | 3.0 | 0.40 | 1.09 | 0.274 |
| AFRM | mr_short_term | short | FAIL | 14 | 0.226 | 0.056 | 2.021 | 4.73 | 0.83 | 0.6 | 1.00 | 1.12 | 0.226 |
| HOOD | mom_12_1 | long | FAIL | 20 | 0.261 | -0.146 | 1.066 | 1.81 | 0.58 | 3.0 | 0.67 | 1.11 | 0.213 |
| SMCI | mom_12_1 | long | FAIL | 22 | 0.245 | -0.198 | 0.972 | 1.80 | 0.52 | 2.2 | 0.60 | 0.52 | 0.190 |
| FSLY | mom_12_1 | short | FAIL | 14 | 0.188 | -0.090 | 1.175 | 2.22 | 0.77 | 2.1 | 1.00 | 2.09 | 0.188 |
| JPM | low_vol_trend | long | FAIL | 24 | 0.262 | -0.146 | 0.968 | 1.61 | 0.58 | 3.4 | 0.50 | 0.40 | 0.185 |
| PTON | mom_12_1 | short | FAIL | 20 | 0.228 | -0.028 | 1.385 | 2.22 | 0.65 | 1.4 | 0.60 | 0.55 | 0.176 |
| TDOC | mom_12_1 | short | FAIL | 18 | 0.247 | 0.011 | 1.745 | 2.66 | 0.72 | 1.1 | 0.50 | 0.64 | 0.175 |
| NFLX | mr_short_term | long | FAIL | 23 | 0.221 | 0.028 | 1.827 | 3.35 | 0.68 | 1.2 | 0.60 | 0.29 | 0.171 |
| DVN | mom_12_1 | long | FAIL | 12 | 0.188 | -0.305 | 0.555 | 1.43 | 0.50 | 2.0 | 0.75 | 1.00 | 0.163 |
| ZTO | overnight_hold | short | FAIL | 31 | 0.174 | 0.036 | 1.872 | 3.47 | 0.63 | 0.6 | 0.83 | 0.91 | 0.159 |
| PYPL | mr_short_term | short | FAIL | 18 | 0.177 | 0.042 | 2.175 | 4.32 | 0.77 | 0.7 | 0.75 | 0.68 | 0.153 |
| ROKU | mom_12_1 | short | FAIL | 13 | 0.174 | -0.147 | 0.844 | 1.93 | 0.62 | 2.0 | 0.75 | 0.83 | 0.151 |
| PLUG | mr_short_term | short | FAIL | 26 | 0.139 | -0.066 | 1.276 | 1.86 | 0.58 | 2.2 | 0.75 | 1.20 | 0.121 |
| XLF | low_vol_trend | long | FAIL | 24 | 0.171 | -0.408 | 0.610 | 1.34 | 0.50 | 3.9 | 0.40 | 2.18 | 0.108 |
| SMR | overnight_hold | long | FAIL | 29 | 0.114 | 0.038 | 2.482 | 3.32 | 0.80 | 0.4 | 0.75 | 0.75 | 0.099 |
| MSFT | low_vol_trend | long | FAIL | 22 | 0.120 | -0.219 | 0.481 | 1.28 | 0.52 | 5.4 | 0.60 | 3.71 | 0.093 |
| BILI | overnight_hold | long | FAIL | 34 | 0.092 | 0.022 | 2.184 | 2.71 | 0.63 | 0.4 | 1.00 | 1.39 | 0.092 |
| ANET | overnight_hold | short | FAIL | 28 | 0.102 | -0.003 | 1.268 | 3.03 | 0.64 | 0.8 | 0.80 | 1.43 | 0.091 |
| ZIM | overnight_hold | long | FAIL | 26 | 0.091 | 0.031 | 2.549 | 4.22 | 0.74 | 0.3 | 1.00 | 1.18 | 0.091 |
| DAL | overnight_hold | long | FAIL | 28 | 0.099 | 0.023 | 2.023 | 2.66 | 0.70 | 0.6 | 0.80 | 1.39 | 0.089 |
| NVDA | overnight_hold | long | WATCH | 86 | 0.102 | 0.043 | 2.454 | 2.83 | 0.64 | 1.1 | 0.75 | 1.31 | 0.088 |
| IREN | overnight_hold | long | FAIL | 36 | 0.101 | 0.027 | 2.064 | 3.19 | 0.61 | 0.4 | 0.75 | 0.89 | 0.087 |
| AVGO | mr_short_term | long | FAIL | 26 | 0.121 | -0.048 | 0.938 | 1.67 | 0.62 | 1.8 | 0.50 | 0.86 | 0.086 |
| AI | overnight_hold | long | FAIL | 35 | 0.095 | 0.024 | 1.907 | 2.79 | 0.67 | 0.7 | 0.75 | 0.90 | 0.083 |

## Pooled out-of-sample by family

| family | side | n | exp_r | pf | wr |
|---|---|---|---|---|---|
| low_vol_trend | long | 1582 | -0.040 | 0.92 | 0.41 |
| mom_12_1 | long | 1146 | 0.136 | 1.38 | 0.49 |
| mom_12_1 | short | 1220 | -0.070 | 0.83 | 0.49 |
| mr_20d | long | 789 | 0.110 | 1.29 | 0.51 |
| mr_20d | short | 905 | -0.114 | 0.79 | 0.41 |
| mr_short_term | long | 3249 | 0.005 | 1.03 | 0.50 |
| mr_short_term | short | 4187 | 0.021 | 1.12 | 0.52 |
| overnight_hold | long | 15002 | -0.002 | 0.97 | 0.49 |
| overnight_hold | short | 17138 | -0.003 | 0.95 | 0.49 |
| rs_rotation | long | 1359 | 0.046 | 1.13 | 0.47 |

## Per-regime expectancy by family

| family | side | mixed | risk_off | risk_on |
|---|---|---|---|---|
| low_vol_trend | long | -0.076 (n=1487) | 0.457 (n=74) | 0.749 (n=21) |
| mom_12_1 | long | 0.063 (n=1047) | 0.789 (n=73) | 1.271 (n=26) |
| mom_12_1 | short | -0.041 (n=1110) | -0.579 (n=72) | 0.038 (n=38) |
| mr_20d | long | 0.055 (n=530) | 0.215 (n=248) | 0.388 (n=11) |
| mr_20d | short | -0.122 (n=856) | 0.342 (n=38) | -1.008 (n=11) |
| mr_short_term | long | 0.004 (n=2582) | 0.012 (n=610) | -0.027 (n=57) |
| mr_short_term | short | 0.011 (n=3836) | 0.083 (n=95) | 0.147 (n=256) |
| overnight_hold | long | 0.001 (n=8894) | -0.026 (n=3741) | 0.022 (n=2367) |
| overnight_hold | short | -0.009 (n=10908) | 0.003 (n=377) | 0.008 (n=5853) |
| rs_rotation | long | 0.021 (n=1176) | 0.241 (n=128) | 0.131 (n=55) |

## Pooled rows vs baseline

| family | side | regime | baseline | n | exp_R | baseline exp_R | baseline n | replayed | excess | t_session | recovery | verdict | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mom_12_1 | long | risk_on | selection | 26 | 1.271 | 0.103 | 12 | 0.462 | 1.168 | 1.76 | 6.17 | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session, baseline_population_mismatch, baseline_underreplayed |
| mom_12_1 | long | risk_off | selection | 73 | 0.789 | 0.158 | 54 | 0.740 | 0.632 | 1.31 | 8.08 | FAIL | min_n, min_sessions, min_t_stat_session, baseline_underreplayed |
| mr_20d | short | risk_off | selection | 38 | 0.342 | -0.066 | 23 | 0.605 | 0.408 | 1.18 | 2.23 | FAIL | min_n, min_sessions, min_recovery, min_t_stat, min_t_stat_session, baseline_underreplayed |
| mr_20d | long | risk_on | selection | 11 | 0.388 | 0.134 | 5 | 0.455 | 0.254 | 0.84 | 1.17 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch, baseline_underreplayed |
| mom_12_1 | short | risk_on | selection | 38 | 0.038 | -0.211 | 17 | 0.447 | 0.249 | -0.06 | 0.17 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch, baseline_underreplayed |
| mr_short_term | short | risk_off | selection | 95 | 0.083 | -0.093 | 62 | 0.653 | 0.176 | 2.20 | 3.16 | FAIL | min_n, min_sessions, min_t_stat, min_t_stat_session, baseline_underreplayed |
| rs_rotation | long | risk_off | selection | 128 | 0.241 | 0.128 | 93 | 0.727 | 0.114 | 1.04 | 2.57 | FAIL | min_n, min_sessions, min_recovery, min_decay, min_t_stat, min_t_stat_session, baseline_underreplayed |
| mr_20d | long | risk_off | selection | 248 | 0.215 | 0.103 | 177 | 0.714 | 0.112 | 1.67 | 2.45 | WATCH | min_n, min_recovery, min_stability, min_t_stat, min_t_stat_session, baseline_underreplayed |
| mr_short_term | long | risk_off | selection | 610 | 0.012 | -0.060 | 421 | 0.690 | 0.073 | 0.19 | 0.22 | WATCH | min_pf, min_exp_r, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_underreplayed |
| mr_20d | long | mixed | selection | 530 | 0.055 | -0.017 | 391 | 0.738 | 0.072 | -0.53 | 1.01 | WATCH | min_pf, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_underreplayed |
| mr_short_term | short | risk_on | selection | 256 | 0.147 | 0.099 | 194 | 0.758 | 0.048 | -0.56 | 2.33 | WATCH | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r, baseline_underreplayed |
| rs_rotation | long | mixed | selection | 1176 | 0.021 | -0.016 | 806 | 0.685 | 0.037 | 0.74 | 0.31 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| mom_12_1 | long | mixed | selection | 1047 | 0.063 | 0.029 | 710 | 0.678 | 0.033 | 1.42 | 0.73 | WATCH | min_pf, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_excess_r, baseline_underreplayed |
| rs_rotation | long | risk_on | selection | 55 | 0.131 | 0.100 | 32 | 0.582 | 0.031 | 0.29 | 0.69 | FAIL | min_n, min_sessions, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| overnight_hold | long | risk_off | selection | 3741 | -0.026 | -0.040 | 2497 | 0.667 | 0.015 | -0.46 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| mr_short_term | short | mixed | selection | 3836 | 0.011 | 0.001 | 2771 | 0.722 | 0.010 | -1.34 | 0.43 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| overnight_hold | short | risk_on | selection | 5853 | 0.008 | 0.002 | 3535 | 0.604 | 0.006 | -0.69 | 0.63 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat_session, min_excess_r, baseline_underreplayed |
| low_vol_trend | long | mixed | selection | 1487 | -0.076 | -0.080 | 1413 | 0.950 | 0.004 | 0.23 | 0.00 | FAIL | min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | long | risk_on | selection | 2367 | 0.022 | 0.018 | 1318 | 0.557 | 0.004 | 0.65 | 3.49 | WATCH | min_exp_r, min_stability, min_decay, min_t_stat_session, min_excess_r, baseline_underreplayed |
| overnight_hold | long | mixed | selection | 8894 | 0.001 | -0.002 | 5440 | 0.612 | 0.004 | 2.00 | 0.10 | FAIL | min_pf, min_exp_r, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| overnight_hold | short | mixed | selection | 10908 | -0.009 | -0.012 | 6471 | 0.593 | 0.002 | -2.69 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| overnight_hold | short | risk_off | selection | 377 | 0.003 | 0.005 | 226 | 0.599 | -0.002 | -0.28 | 0.11 | FAIL | min_n, min_pf, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| mr_short_term | long | mixed | selection | 2582 | 0.004 | 0.049 | 1791 | 0.694 | -0.044 | 0.74 | 0.19 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| mom_12_1 | short | mixed | selection | 1110 | -0.041 | 0.004 | 790 | 0.712 | -0.045 | -1.67 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| mr_short_term | long | risk_on | selection | 57 | -0.027 | 0.018 | 37 | 0.649 | -0.045 | 0.48 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| mr_20d | short | mixed | selection | 856 | -0.122 | -0.011 | 647 | 0.756 | -0.111 | -1.96 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| low_vol_trend | long | risk_off | selection | 74 | 0.457 | 0.584 | 70 | 0.946 | -0.128 | 2.01 | 1.89 | FAIL | min_n, min_sessions, min_recovery, min_t_stat, min_t_stat_session, min_excess_r |
| mom_12_1 | short | risk_off | selection | 72 | -0.579 | -0.164 | 52 | 0.722 | -0.415 | -2.00 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_underreplayed |
| low_vol_trend | long | risk_on | selection | 21 | 0.749 | 1.419 | 21 | 1.000 | -0.670 | 0.16 | 5.28 | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| mr_20d | short | risk_on | selection | 11 | -1.008 | 0.095 | 5 | 0.455 | -1.102 | -4.04 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_population_mismatch, baseline_underreplayed |

## Search accounting

| family | grid combos × sides | unconditional rows scored | PASS | WATCH |
|---|---|---|---|---|
| low_vol_trend | 2 × 1 = 2 | 238 | 0 | 0 |
| mom_12_1 | 2 × 2 = 4 | 424 | 0 | 0 |
| mr_20d | 12 × 2 = 24 | 885 | 0 | 0 |
| mr_short_term | 6 × 2 = 12 | 961 | 0 | 0 |
| overnight_hold | 8 × 2 = 16 | 971 | 0 | 43 |
| rs_rotation | 4 × 1 = 4 | 300 | 0 | 0 |
| **total** | 62 | 3779 | 0 | 43 |

6 families, 62 grid points × sides evaluated, 0 PASS

## Controls

_no control runs found_ (real PASS rows: 0)

## Symbols with no passing lane

530 of 530 symbols have no PASS lane.

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

