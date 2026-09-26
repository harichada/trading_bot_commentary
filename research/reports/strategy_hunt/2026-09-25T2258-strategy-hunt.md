# Strategy hunt report

Computed: 2026-09-25T22:58:39. Rows: 12764. Symbols: 539.


## Verdicts by family

| family | side | PASS | WATCH | FAIL | coldstart WATCH |
|---|---|---|---|---|---|
| dt_momentum | long | 0 | 35 | 502 | 0 |
| dt_momentum | short | 0 | 13 | 525 | 0 |
| gap_play | long | 0 | 18 | 501 | 0 |
| gap_play | short | 0 | 33 | 487 | 0 |
| mean_rev_intraday | long | 0 | 3 | 527 | 0 |
| mr_20d | long | 0 | 0 | 497 | 0 |
| mr_20d | short | 0 | 0 | 513 | 0 |
| mr_short_term | long | 0 | 0 | 510 | 0 |
| mr_short_term | short | 0 | 0 | 509 | 0 |
| multi_day_drop | long | 0 | 0 | 511 | 0 |
| news_catalyst | long | 0 | 0 | 351 | 0 |
| news_catalyst | short | 0 | 0 | 256 | 0 |
| orb | long | 0 | 0 | 509 | 0 |
| orb | short | 0 | 0 | 498 | 0 |
| orb_inplay | long | 0 | 0 | 513 | 0 |
| orb_inplay | short | 0 | 0 | 510 | 0 |
| overnight_hold | long | 0 | 39 | 469 | 0 |
| overnight_hold | short | 0 | 29 | 477 | 0 |
| pead_smallcap | long | 0 | 0 | 177 | 0 |
| pead_smallcap | short | 0 | 0 | 102 | 0 |
| pead_wide | long | 0 | 0 | 389 | 0 |
| pead_wide | short | 0 | 0 | 260 | 0 |
| post_news_drift | long | 0 | 0 | 312 | 0 |
| post_news_drift | short | 0 | 0 | 333 | 0 |
| pullback_trend | long | 0 | 0 | 481 | 0 |
| rs_leader | long | 0 | 0 | 169 | 0 |
| rs_rotation | long | 0 | 0 | 344 | 0 |
| vol_breakout | long | 0 | 0 | 131 | 0 |
| vol_breakout | short | 0 | 0 | 164 | 0 |
| vwap_reversion | long | 0 | 10 | 523 | 0 |
| vwap_reversion | short | 0 | 15 | 519 | 0 |

## Top lanes by score

| symbol | family | side | verdict | n | exp_r | exp_r_lb | t_stat | pf | wr | max_dd_r | stability | decay | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RIVN | orb_inplay | long | FAIL | 48 | 1.070 | 0.297 | 2.358 | 2.24 | 0.42 | 8.7 | 1.00 | 0.57 | 1.070 |
| NBIS | orb_inplay | long | FAIL | 37 | 0.980 | -0.023 | 1.573 | 2.00 | 0.32 | 8.8 | 0.75 | 0.35 | 0.849 |
| ZSL | orb_inplay | short | FAIL | 27 | 0.689 | -0.657 | 0.727 | 1.56 | 0.26 | 16.3 | 0.80 | 1.16 | 0.616 |
| MSTR | rs_rotation | long | FAIL | 23 | 0.687 | -0.004 | 1.683 | 2.91 | 0.57 | 4.1 | 0.80 | 1.95 | 0.614 |
| CZR | dt_momentum | long | WATCH | 110 | 0.569 | 0.275 | 3.260 | 1.99 | 0.50 | 8.3 | 1.00 | 1.22 | 0.569 |
| IBN | gap_play | short | FAIL | 23 | 0.557 | 0.168 | 2.199 | 3.00 | 0.59 | 3.4 | 1.00 | 1.24 | 0.557 |
| BIDU | orb_inplay | long | FAIL | 17 | 0.756 | -0.246 | 1.116 | 2.01 | 0.41 | 7.6 | 0.50 | 1.25 | 0.535 |
| LCID | gap_play | short | FAIL | 35 | 0.473 | 0.048 | 1.858 | 1.91 | 0.54 | 5.4 | 1.00 | 1.57 | 0.473 |
| BE | rs_rotation | long | FAIL | 22 | 0.535 | 0.002 | 1.560 | 2.46 | 0.62 | 3.4 | 0.75 | 0.28 | 0.463 |
| CVNA | rs_rotation | long | FAIL | 26 | 0.433 | -0.002 | 1.463 | 2.32 | 0.55 | 2.2 | 1.00 | 2.02 | 0.433 |
| SMH | gap_play | short | FAIL | 28 | 0.466 | 0.086 | 1.900 | 2.24 | 0.59 | 4.5 | 0.75 | 1.17 | 0.404 |
| TSDD | gap_play | short | FAIL | 33 | 0.381 | 0.017 | 1.429 | 1.68 | 0.48 | 3.4 | 1.00 | 1.14 | 0.381 |
| LIFE | gap_play | short | FAIL | 56 | 0.378 | 0.086 | 1.835 | 1.67 | 0.55 | 4.1 | 0.88 | 0.10 | 0.354 |
| RGTI | gap_play | long | FAIL | 26 | 0.433 | -0.005 | 1.450 | 1.81 | 0.54 | 7.5 | 0.60 | 2.27 | 0.336 |
| NBIS | vwap_reversion | long | FAIL | 37 | 0.385 | -0.017 | 1.615 | 1.90 | 0.58 | 4.8 | 0.75 | 1.15 | 0.334 |
| CZR | dt_momentum | short | WATCH | 141 | 0.370 | 0.146 | 2.435 | 1.58 | 0.44 | 17.6 | 0.75 | 1.08 | 0.320 |
| NTRA | gap_play | long | FAIL | 44 | 0.320 | -0.029 | 1.526 | 1.68 | 0.56 | 5.5 | 1.00 | 0.83 | 0.320 |
| AMDL | mean_rev_intraday | long | FAIL | 16 | 0.369 | -0.275 | 0.894 | 1.61 | 0.50 | 4.5 | 0.75 | 1.27 | 0.320 |
| TSCO | dt_momentum | long | WATCH | 72 | 0.305 | 0.102 | 2.115 | 1.75 | 0.58 | 6.4 | 1.00 | 1.04 | 0.305 |
| BEKE | mean_rev_intraday | long | FAIL | 24 | 0.389 | -0.067 | 1.251 | 1.72 | 0.50 | 3.1 | 0.60 | 1.85 | 0.302 |
| TSDD | vwap_reversion | short | FAIL | 33 | 0.382 | -0.072 | 1.277 | 1.74 | 0.55 | 4.9 | 0.60 | 0.55 | 0.296 |
| SMH | gap_play | long | FAIL | 19 | 0.341 | -0.070 | 1.418 | 2.28 | 0.68 | 2.2 | 0.75 | 1.56 | 0.295 |
| F | gap_play | long | FAIL | 29 | 0.381 | -0.048 | 1.386 | 1.73 | 0.52 | 4.6 | 0.60 | 0.41 | 0.295 |
| LIFE | vwap_reversion | short | FAIL | 57 | 0.335 | 0.032 | 1.635 | 1.65 | 0.54 | 6.0 | 0.75 | 0.27 | 0.290 |
| MCHP | gap_play | short | FAIL | 38 | 0.334 | 0.028 | 1.679 | 1.87 | 0.53 | 3.7 | 0.75 | 0.97 | 0.289 |
| SPOT | vwap_reversion | long | FAIL | 48 | 0.303 | -0.049 | 1.207 | 1.50 | 0.46 | 7.0 | 0.88 | 1.38 | 0.284 |
| NTRA | vwap_reversion | short | WATCH | 61 | 0.321 | -0.060 | 1.317 | 1.51 | 0.47 | 8.5 | 0.75 | 0.72 | 0.278 |
| ANET | gap_play | long | FAIL | 32 | 0.276 | -0.010 | 1.587 | 1.97 | 0.61 | 2.1 | 1.00 | 1.37 | 0.276 |
| SOUN | gap_play | long | FAIL | 53 | 0.345 | 0.004 | 1.668 | 1.62 | 0.49 | 6.6 | 0.60 | 1.25 | 0.267 |
| NCLH | mean_rev_intraday | long | FAIL | 45 | 0.327 | -0.047 | 1.482 | 1.59 | 0.49 | 4.3 | 0.67 | 1.22 | 0.267 |

## Pooled out-of-sample by family

| family | side | n | exp_r | pf | wr |
|---|---|---|---|---|---|
| dt_momentum | long | 71480 | -0.151 | 0.76 | 0.38 |
| dt_momentum | short | 84611 | -0.203 | 0.69 | 0.37 |
| gap_play | long | 21246 | -0.066 | 0.90 | 0.39 |
| gap_play | short | 21869 | -0.037 | 0.94 | 0.39 |
| mean_rev_intraday | long | 10094 | -0.173 | 0.78 | 0.35 |
| mr_20d | long | 1047 | 0.091 | 1.23 | 0.51 |
| mr_20d | short | 1227 | -0.184 | 0.70 | 0.41 |
| mr_short_term | long | 4875 | 0.003 | 1.02 | 0.50 |
| mr_short_term | short | 3034 | 0.009 | 1.05 | 0.51 |
| multi_day_drop | long | 2822 | 0.004 | 1.02 | 0.52 |
| news_catalyst | long | 1326 | -0.064 | 0.88 | 0.41 |
| news_catalyst | short | 601 | -0.006 | 0.99 | 0.44 |
| orb | long | 2423 | -0.300 | 0.65 | 0.34 |
| orb | short | 2331 | -0.260 | 0.69 | 0.37 |
| orb_inplay | long | 5084 | -0.167 | 0.87 | 0.25 |
| orb_inplay | short | 4991 | -0.226 | 0.83 | 0.25 |
| overnight_hold | long | 27698 | -0.007 | 0.90 | 0.48 |
| overnight_hold | short | 32614 | -0.012 | 0.82 | 0.45 |
| pead_smallcap | long | 177 | -0.007 | 0.99 | 0.42 |
| pead_smallcap | short | 102 | -0.141 | 0.79 | 0.42 |
| pead_wide | long | 666 | -0.022 | 0.96 | 0.49 |
| pead_wide | short | 352 | -0.138 | 0.78 | 0.47 |
| post_news_drift | long | 479 | -0.031 | 0.93 | 0.46 |
| post_news_drift | short | 490 | 0.030 | 1.09 | 0.52 |
| pullback_trend | long | 876 | -0.021 | 0.95 | 0.46 |
| rs_leader | long | 169 | -0.085 | 0.84 | 0.41 |
| rs_rotation | long | 1893 | 0.090 | 1.25 | 0.47 |
| vol_breakout | long | 150 | -0.109 | 0.81 | 0.39 |
| vol_breakout | short | 221 | -0.125 | 0.72 | 0.43 |
| vwap_reversion | long | 53165 | -0.210 | 0.67 | 0.38 |
| vwap_reversion | short | 51336 | -0.252 | 0.62 | 0.37 |

## Per-regime expectancy by family

| family | side | mixed | risk_off | risk_on |
|---|---|---|---|---|
| dt_momentum | long | -0.157 (n=50997) | -0.140 (n=8020) | -0.130 (n=12463) |
| dt_momentum | short | -0.176 (n=61118) | -0.275 (n=8868) | -0.276 (n=14625) |
| gap_play | long | -0.119 (n=15864) | 0.060 (n=2466) | 0.114 (n=2916) |
| gap_play | short | -0.006 (n=17749) | -0.152 (n=2477) | -0.191 (n=1643) |
| mean_rev_intraday | long | -0.187 (n=8252) | -0.090 (n=1299) | -0.159 (n=543) |
| mr_20d | long | 0.027 (n=701) | 0.227 (n=331) | 0.034 (n=15) |
| mr_20d | short | -0.215 (n=1150) | 0.430 (n=60) | -0.242 (n=17) |
| mr_short_term | long | -0.002 (n=3843) | 0.023 (n=944) | -0.016 (n=88) |
| mr_short_term | short | 0.003 (n=2804) | 0.088 (n=83) | 0.077 (n=147) |
| multi_day_drop | long | 0.004 (n=2381) | -0.009 (n=390) | 0.113 (n=51) |
| news_catalyst | long | -0.068 (n=945) | -0.153 (n=123) | -0.008 (n=258) |
| news_catalyst | short | 0.044 (n=424) | -0.225 (n=96) | -0.005 (n=81) |
| orb | long | -0.323 (n=1750) | -0.380 (n=195) | -0.185 (n=478) |
| orb | short | -0.258 (n=1826) | -0.196 (n=339) | -0.407 (n=166) |
| orb_inplay | long | -0.182 (n=4331) | -0.019 (n=447) | -0.172 (n=306) |
| orb_inplay | short | -0.254 (n=4300) | 0.003 (n=463) | -0.176 (n=228) |
| overnight_hold | long | -0.004 (n=16218) | -0.024 (n=7014) | 0.009 (n=4466) |
| overnight_hold | short | -0.017 (n=20956) | -0.005 (n=862) | -0.003 (n=10796) |
| pead_smallcap | long | -0.075 (n=133) | 1.062 (n=17) | -0.342 (n=27) |
| pead_smallcap | short | -0.176 (n=76) | 0.431 (n=14) | -0.590 (n=12) |
| pead_wide | long | -0.060 (n=521) | 0.338 (n=49) | -0.002 (n=96) |
| pead_wide | short | 0.018 (n=290) | -1.957 (n=30) | 0.150 (n=32) |
| post_news_drift | long | -0.004 (n=443) | -0.290 (n=27) | -0.611 (n=9) |
| post_news_drift | short | 0.052 (n=394) | -0.132 (n=81) | 0.338 (n=15) |
| pullback_trend | long | -0.029 (n=840) | 0.016 (n=31) | 1.010 (n=5) |
| rs_leader | long | -0.066 (n=160) | -0.419 (n=9) |  |
| rs_rotation | long | 0.078 (n=1623) | 0.197 (n=179) | 0.091 (n=91) |
| vol_breakout | long | -0.095 (n=146) | -0.633 (n=4) |  |
| vol_breakout | short | -0.174 (n=179) | 0.071 (n=41) | 0.629 (n=1) |
| vwap_reversion | long | -0.242 (n=42095) | -0.058 (n=7673) | -0.157 (n=3397) |
| vwap_reversion | short | -0.227 (n=29979) | -0.182 (n=4459) | -0.315 (n=16898) |

## Pooled rows vs shuffled baseline

| family | side | regime | n | exp_R | baseline exp_R | baseline n | excess | t_session | recovery | verdict | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| orb_inplay | long | risk_on | 306 | -0.172 | -4.449 | 290 | 4.277 | -0.86 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| orb_inplay | long | risk_off | 447 | -0.019 | -1.413 | 479 | 1.394 | -0.59 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| orb_inplay | long | mixed | 4331 | -0.182 | -1.538 | 4706 | 1.355 | -3.63 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| orb_inplay | short | mixed | 4300 | -0.254 | -1.562 | 4527 | 1.308 | -3.50 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| pead_smallcap | long | risk_off | 17 | 1.062 | 0.046 | 28 | 1.016 | 1.62 | 6.27 | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session |
| orb_inplay | short | risk_off | 463 | 0.003 | -0.800 | 520 | 0.803 | -0.54 | 0.01 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| orb_inplay | short | risk_on | 228 | -0.176 | -0.959 | 281 | 0.784 | -1.71 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| vol_breakout | long | risk_off | 4 | -0.633 | -1.070 | 4 | 0.436 | -1.38 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session |
| post_news_drift | short | risk_on | 15 | 0.338 | 0.072 | 15 | 0.267 | 1.51 | 5.03 | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session |
| pead_smallcap | short | risk_off | 14 | 0.431 | 0.222 | 9 | 0.209 | 1.11 | 2.91 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| dt_momentum | short | risk_on | 14625 | -0.276 | -0.451 | 31834 | 0.176 | -4.59 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch |
| gap_play | long | risk_off | 2466 | 0.060 | -0.094 | 2666 | 0.154 | -1.87 | 0.71 | WATCH | min_pf, min_recovery, min_stability, min_t_stat, min_t_stat_session |
| mean_rev_intraday | long | risk_off | 1299 | -0.090 | -0.242 | 599 | 0.151 | -1.43 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch |
| mr_20d | short | risk_off | 60 | 0.430 | 0.291 | 60 | 0.139 | 1.04 | 4.10 | FAIL | min_n, min_sessions, min_t_stat, min_t_stat_session |
| dt_momentum | long | risk_off | 8020 | -0.140 | -0.269 | 15176 | 0.129 | -2.22 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| orb | short | mixed | 1826 | -0.258 | -0.383 | 9353 | 0.125 | -8.73 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch |
| dt_momentum | long | mixed | 50997 | -0.157 | -0.265 | 92489 | 0.108 | -8.28 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| pead_wide | short | risk_on | 32 | 0.150 | 0.045 | 32 | 0.105 | 0.56 | 0.95 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb |
| gap_play | long | risk_on | 2916 | 0.114 | 0.016 | 2298 | 0.098 | -0.70 | 2.06 | WATCH | min_wr, min_recovery, min_stability, min_t_stat_session |
| orb | short | risk_on | 166 | -0.407 | -0.499 | 1027 | 0.092 | -3.42 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch |
| vwap_reversion | short | risk_on | 16898 | -0.315 | -0.399 | 16194 | 0.084 | -5.71 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| orb | short | risk_off | 339 | -0.196 | -0.280 | 935 | 0.084 | -3.12 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, baseline_population_mismatch |
| gap_play | short | mixed | 17749 | -0.006 | -0.077 | 17016 | 0.071 | -3.94 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| vwap_reversion | long | risk_off | 7673 | -0.058 | -0.126 | 7033 | 0.067 | 1.20 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb |
| gap_play | short | risk_on | 1643 | -0.191 | -0.235 | 2095 | 0.044 | -3.11 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | short | risk_on | 147 | 0.077 | 0.036 | 147 | 0.041 | -1.11 | 2.73 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| post_news_drift | long | risk_off | 27 | -0.290 | -0.325 | 27 | 0.034 | -1.15 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vol_breakout | long | mixed | 146 | -0.095 | -0.126 | 146 | 0.031 | -0.89 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vol_breakout | short | risk_off | 41 | 0.071 | 0.046 | 41 | 0.025 | -0.58 | 0.50 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| gap_play | long | mixed | 15864 | -0.119 | -0.140 | 17043 | 0.021 | -5.45 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| gap_play | short | risk_off | 2477 | -0.152 | -0.173 | 2320 | 0.021 | -2.21 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| orb | long | risk_on | 478 | -0.185 | -0.203 | 1374 | 0.018 | -3.09 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_population_mismatch |
| pead_wide | short | mixed | 290 | 0.018 | 0.001 | 290 | 0.017 | 0.60 | 0.15 | FAIL | min_n, min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_rotation | long | risk_off | 179 | 0.197 | 0.181 | 179 | 0.016 | 0.98 | 2.80 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_excess_r |
| dt_momentum | short | mixed | 61118 | -0.176 | -0.192 | 95960 | 0.016 | -13.31 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | short | risk_off | 83 | 0.088 | 0.073 | 83 | 0.015 | 1.55 | 3.46 | FAIL | min_n, min_sessions, min_t_stat, min_t_stat_session, min_excess_r |
| post_news_drift | long | mixed | 443 | -0.004 | -0.017 | 443 | 0.013 | -0.08 | 0.00 | FAIL | min_n, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_20d | short | mixed | 1150 | -0.215 | -0.224 | 1150 | 0.010 | -2.56 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| post_news_drift | long | risk_on | 9 | -0.611 | -0.621 | 9 | 0.009 | -1.26 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| mr_20d | long | mixed | 701 | 0.027 | 0.019 | 701 | 0.008 | -0.64 | 0.40 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | short | risk_off | 862 | -0.005 | -0.011 | 858 | 0.006 | -2.00 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | short | mixed | 2804 | 0.003 | -0.001 | 2804 | 0.003 | -0.95 | 0.08 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_rotation | long | mixed | 1623 | 0.078 | 0.075 | 1623 | 0.003 | 0.80 | 1.65 | WATCH | min_recovery, min_stability, min_t_stat, min_t_stat_session, min_excess_r |
| overnight_hold | long | risk_on | 4466 | 0.009 | 0.006 | 4475 | 0.002 | -0.64 | 1.02 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| pead_smallcap | long | mixed | 133 | -0.075 | -0.077 | 140 | 0.002 | -0.30 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | long | risk_off | 944 | 0.023 | 0.022 | 942 | 0.001 | 0.92 | 0.52 | WATCH | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vol_breakout | short | risk_on | 1 | 0.629 | 0.629 | 1 | 0.000 | 0.00 | inf | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| post_news_drift | short | mixed | 394 | 0.052 | 0.052 | 394 | -0.000 | 0.88 | 0.97 | WATCH | min_n, min_pf, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_leader | long | risk_off | 9 | -0.419 | -0.418 | 9 | -0.001 | -1.04 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| overnight_hold | short | mixed | 20956 | -0.017 | -0.016 | 20919 | -0.001 | -4.34 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pullback_trend | long | risk_on | 5 | 1.010 | 1.011 | 5 | -0.001 | 2.92 | inf | FAIL | min_n, min_sessions, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| overnight_hold | long | mixed | 16218 | -0.004 | -0.003 | 16131 | -0.001 | 0.72 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | long | risk_off | 7014 | -0.024 | -0.023 | 6981 | -0.001 | -0.42 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | short | risk_on | 10796 | -0.003 | -0.001 | 10776 | -0.002 | -2.38 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | long | mixed | 3843 | -0.002 | 0.002 | 3844 | -0.004 | 0.52 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| multi_day_drop | long | mixed | 2381 | 0.004 | 0.015 | 2381 | -0.010 | 0.29 | 0.15 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | long | mixed | 521 | -0.060 | -0.049 | 521 | -0.011 | -1.04 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pullback_trend | long | mixed | 840 | -0.029 | -0.018 | 840 | -0.011 | 0.41 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| multi_day_drop | long | risk_off | 390 | -0.009 | 0.003 | 390 | -0.012 | -0.13 | 0.00 | FAIL | min_n, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vol_breakout | short | mixed | 179 | -0.174 | -0.157 | 179 | -0.017 | -2.28 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| news_catalyst | long | mixed | 945 | -0.068 | -0.049 | 977 | -0.019 | -1.38 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| post_news_drift | short | risk_off | 81 | -0.132 | -0.112 | 81 | -0.021 | -0.96 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_rotation | long | risk_on | 91 | 0.091 | 0.113 | 91 | -0.022 | 0.01 | 0.95 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | long | risk_on | 88 | -0.016 | 0.010 | 88 | -0.025 | -0.12 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| news_catalyst | short | mixed | 424 | 0.044 | 0.069 | 438 | -0.026 | 0.32 | 1.22 | WATCH | min_n, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | long | risk_on | 96 | -0.002 | 0.025 | 96 | -0.027 | -0.84 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_20d | long | risk_off | 331 | 0.227 | 0.258 | 331 | -0.031 | 1.71 | 2.08 | WATCH | min_n, min_recovery, min_stability, min_t_stat_session, min_excess_r |
| orb | long | mixed | 1750 | -0.323 | -0.291 | 9286 | -0.032 | -8.74 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_population_mismatch |
| mr_20d | short | risk_on | 17 | -0.242 | -0.207 | 17 | -0.035 | -1.22 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_leader | long | mixed | 160 | -0.066 | -0.028 | 160 | -0.038 | -1.09 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| dt_momentum | short | risk_off | 8868 | -0.275 | -0.230 | 13881 | -0.045 | -3.97 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| news_catalyst | short | risk_on | 81 | -0.005 | 0.043 | 94 | -0.047 | 0.29 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| multi_day_drop | long | risk_on | 51 | 0.113 | 0.163 | 51 | -0.050 | 0.90 | 2.08 | FAIL | min_n, min_sessions, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vwap_reversion | short | mixed | 29979 | -0.227 | -0.174 | 42338 | -0.054 | -12.11 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vwap_reversion | long | mixed | 42095 | -0.242 | -0.188 | 47032 | -0.054 | -5.97 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pullback_trend | long | risk_off | 31 | 0.016 | 0.092 | 31 | -0.075 | 0.39 | 0.13 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mean_rev_intraday | long | mixed | 8252 | -0.187 | -0.102 | 4225 | -0.084 | -2.93 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_smallcap | short | mixed | 76 | -0.176 | -0.085 | 88 | -0.091 | -0.97 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| news_catalyst | long | risk_on | 258 | -0.008 | 0.099 | 251 | -0.107 | -0.77 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vwap_reversion | short | risk_off | 4459 | -0.182 | -0.067 | 5709 | -0.114 | -2.46 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| dt_momentum | long | risk_on | 12463 | -0.130 | -0.014 | 13555 | -0.115 | -9.17 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | long | risk_off | 49 | 0.338 | 0.453 | 49 | -0.116 | 0.97 | 1.70 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | short | risk_off | 30 | -1.957 | -1.764 | 30 | -0.193 | -1.03 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| news_catalyst | long | risk_off | 123 | -0.153 | 0.045 | 135 | -0.198 | -1.17 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| vwap_reversion | long | risk_on | 3397 | -0.157 | 0.064 | 8592 | -0.221 | -5.29 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_population_mismatch |
| orb | long | risk_off | 195 | -0.380 | -0.153 | 913 | -0.227 | -2.76 | 0.00 | FAIL | min_n, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r, baseline_population_mismatch |
| news_catalyst | short | risk_off | 96 | -0.225 | 0.043 | 96 | -0.268 | -1.84 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_20d | long | risk_on | 15 | 0.034 | 0.355 | 15 | -0.321 | 0.09 | 0.07 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mean_rev_intraday | long | risk_on | 543 | -0.159 | 0.201 | 668 | -0.360 | -2.26 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_smallcap | short | risk_on | 12 | -0.590 | -0.050 | 8 | -0.540 | -0.89 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_smallcap | long | risk_on | 27 | -0.342 | 0.449 | 20 | -0.791 | -0.75 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |

## Search accounting

| family | grid combos × sides | unconditional rows scored | PASS | WATCH |
|---|---|---|---|---|
| dt_momentum | 27 × 2 = 54 | 1075 | 0 | 48 |
| gap_play | 12 × 2 = 24 | 1039 | 0 | 51 |
| mean_rev_intraday | 18 × 1 = 18 | 530 | 0 | 3 |
| mr_20d | 12 × 2 = 24 | 1010 | 0 | 0 |
| mr_short_term | 12 × 2 = 24 | 1019 | 0 | 0 |
| multi_day_drop | 4 × 1 = 4 | 511 | 0 | 0 |
| news_catalyst | 18 × 2 = 36 | 607 | 0 | 0 |
| orb | 12 × 2 = 24 | 1007 | 0 | 0 |
| orb_inplay | 27 × 2 = 54 | 1023 | 0 | 0 |
| overnight_hold | 8 × 2 = 16 | 1014 | 0 | 68 |
| pead_smallcap | 27 × 2 = 54 | 279 | 0 | 0 |
| pead_wide | 9 × 2 = 18 | 649 | 0 | 0 |
| post_news_drift | 18 × 2 = 36 | 645 | 0 | 0 |
| pullback_trend | 12 × 1 = 12 | 481 | 0 | 0 |
| rs_leader | 18 × 1 = 18 | 169 | 0 | 0 |
| rs_rotation | 4 × 1 = 4 | 344 | 0 | 0 |
| vol_breakout | 12 × 2 = 24 | 295 | 0 | 0 |
| vwap_reversion | 18 × 2 = 36 | 1067 | 0 | 25 |
| **total** | 480 | 12764 | 0 | 195 |

18 families, 480 grid points × sides evaluated, 0 PASS

## Controls

- **cost_2.0**: PASS rows 0, mean exp_r -0.256, pooled PASS 0, pooled WATCH 1
- real PASS rows: 0

## Symbols with no passing lane

539 of 539 symbols have no PASS lane.

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
- Ten families times up to 27 parameter sets across hundreds of symbols will produce spurious PASS rows; the in-root shuffled baseline estimates how much of a pooled lane's expectancy is the regime's drift (see "Pooled rows vs shuffled baseline").
- Live indicators are computed on the in-progress bar; research uses closed bars. New lanes must use closed bars.

