# Strategy hunt report

Computed: 2026-09-25T23:05:28. Rows: 3467. Symbols: 531. Holdout: the final 6 months were withheld from this search (re-run with --holdout-months 0 for the confirmation test).


## Verdicts by family

| family | side | PASS | WATCH | FAIL | coldstart WATCH |
|---|---|---|---|---|---|
| mr_short_term | long | 0 | 0 | 496 | 0 |
| mr_short_term | short | 0 | 0 | 506 | 0 |
| multi_day_drop | long | 0 | 0 | 496 | 0 |
| overnight_hold | long | 0 | 31 | 471 | 0 |
| overnight_hold | short | 0 | 27 | 476 | 0 |
| pead_wide | long | 0 | 0 | 385 | 0 |
| pead_wide | short | 0 | 0 | 256 | 0 |
| rs_rotation | long | 0 | 0 | 323 | 0 |

## Top lanes by score

| symbol | family | side | verdict | n | exp_r | exp_r_lb | t_stat | pf | wr | max_dd_r | stability | decay | score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSTR | rs_rotation | long | FAIL | 23 | 0.687 | -0.004 | 1.683 | 2.91 | 0.57 | 4.1 | 0.80 | 1.95 | 0.614 |
| CVNA | rs_rotation | long | FAIL | 23 | 0.507 | -0.042 | 1.531 | 2.45 | 0.57 | 2.2 | 1.00 | 1.73 | 0.507 |
| PLTR | rs_rotation | long | FAIL | 16 | 0.259 | -0.158 | 0.951 | 1.79 | 0.62 | 2.8 | 0.75 | 0.96 | 0.224 |
| AFRM | mr_short_term | short | FAIL | 14 | 0.219 | 0.049 | 1.958 | 4.53 | 0.83 | 0.6 | 1.00 | 1.13 | 0.219 |
| INFY | mr_short_term | short | FAIL | 20 | 0.198 | 0.058 | 2.103 | 3.99 | 0.65 | 0.5 | 0.75 | 1.15 | 0.171 |
| XP | mr_short_term | short | FAIL | 17 | 0.188 | -0.039 | 1.203 | 2.31 | 0.69 | 1.3 | 0.75 | 0.76 | 0.163 |
| APP | rs_rotation | long | FAIL | 28 | 0.184 | -0.137 | 0.979 | 1.61 | 0.63 | 4.4 | 0.75 | 1.23 | 0.160 |
| PYPL | mr_short_term | short | FAIL | 27 | 0.203 | 0.089 | 2.738 | 4.41 | 0.76 | 0.8 | 0.57 | 0.50 | 0.153 |
| SMCI | mr_short_term | long | FAIL | 28 | 0.166 | 0.028 | 1.783 | 3.46 | 0.65 | 0.6 | 0.75 | 0.36 | 0.143 |
| ZTO | overnight_hold | short | FAIL | 43 | 0.168 | 0.030 | 1.448 | 3.85 | 0.64 | 0.6 | 0.67 | 0.28 | 0.137 |
| TSM | overnight_hold | long | FAIL | 37 | 0.122 | 0.028 | 1.698 | 3.75 | 0.67 | 0.5 | 1.00 | 1.31 | 0.122 |
| FXI | overnight_hold | long | WATCH | 73 | 0.130 | 0.056 | 2.567 | 2.35 | 0.58 | 1.9 | 0.88 | 1.47 | 0.121 |
| PLUG | mr_short_term | short | FAIL | 26 | 0.129 | -0.076 | 1.185 | 1.77 | 0.58 | 2.2 | 0.75 | 1.22 | 0.112 |
| NFLX | mr_short_term | long | FAIL | 26 | 0.141 | -0.082 | 1.180 | 2.01 | 0.61 | 1.6 | 0.60 | 0.32 | 0.110 |
| NWL | mr_short_term | short | FAIL | 21 | 0.094 | -0.054 | 1.142 | 1.83 | 0.61 | 1.3 | 1.00 | 0.80 | 0.094 |
| IREN | overnight_hold | long | FAIL | 46 | 0.093 | 0.028 | 2.132 | 2.55 | 0.59 | 0.5 | 1.00 | 0.96 | 0.093 |
| SMR | overnight_hold | long | FAIL | 36 | 0.107 | 0.047 | 2.743 | 3.25 | 0.80 | 0.4 | 0.75 | 0.69 | 0.093 |
| TQQQ | mr_short_term | short | FAIL | 15 | 0.106 | -0.056 | 1.142 | 2.07 | 0.64 | 0.6 | 0.75 | 0.74 | 0.092 |
| PDBC | overnight_hold | long | FAIL | 37 | 0.098 | 0.005 | 1.459 | 1.84 | 0.72 | 1.1 | 0.83 | 0.03 | 0.090 |
| ERIC | mr_short_term | short | FAIL | 22 | 0.102 | -0.066 | 0.922 | 1.74 | 0.50 | 1.3 | 0.75 | 0.71 | 0.089 |
| RXRX | overnight_hold | long | FAIL | 19 | 0.085 | 0.004 | 1.631 | 2.26 | 0.74 | 0.4 | 1.00 | 0.96 | 0.085 |
| AVGO | mr_short_term | long | FAIL | 28 | 0.100 | -0.081 | 0.837 | 1.57 | 0.58 | 1.8 | 0.50 | 0.87 | 0.071 |
| AI | overnight_hold | long | FAIL | 45 | 0.081 | 0.029 | 2.165 | 3.03 | 0.65 | 0.7 | 0.75 | 0.82 | 0.070 |
| NBIS | overnight_hold | long | FAIL | 57 | 0.069 | 0.011 | 2.063 | 2.14 | 0.64 | 0.7 | 1.00 | 1.12 | 0.069 |
| YUMC | overnight_hold | short | FAIL | 40 | 0.092 | -0.057 | 0.761 | 1.95 | 0.50 | 1.3 | 0.57 | 0.94 | 0.069 |
| NVDA | overnight_hold | long | WATCH | 99 | 0.073 | 0.027 | 2.178 | 2.14 | 0.61 | 1.2 | 0.86 | 1.51 | 0.068 |
| IVZ | mr_short_term | short | FAIL | 18 | 0.093 | -0.091 | 0.865 | 1.66 | 0.62 | 1.2 | 0.50 | -0.35 | 0.066 |
| MCHI | overnight_hold | short | FAIL | 47 | 0.077 | -0.056 | 0.827 | 1.66 | 0.57 | 2.2 | 0.71 | 3.24 | 0.065 |
| TOST | overnight_hold | long | FAIL | 25 | 0.073 | 0.019 | 2.081 | 3.05 | 0.70 | 0.5 | 0.75 | 0.81 | 0.063 |
| MSTR | overnight_hold | long | WATCH | 62 | 0.070 | 0.017 | 1.862 | 1.95 | 0.52 | 1.2 | 0.75 | 0.14 | 0.061 |

## Pooled out-of-sample by family

| family | side | n | exp_r | pf | wr |
|---|---|---|---|---|---|
| mr_short_term | long | 4409 | 0.001 | 1.01 | 0.50 |
| mr_short_term | short | 5601 | 0.020 | 1.11 | 0.52 |
| multi_day_drop | long | 2537 | 0.007 | 1.03 | 0.52 |
| overnight_hold | long | 25423 | -0.009 | 0.88 | 0.47 |
| overnight_hold | short | 29900 | -0.013 | 0.81 | 0.45 |
| pead_wide | long | 657 | -0.023 | 0.96 | 0.49 |
| pead_wide | short | 345 | -0.144 | 0.78 | 0.47 |
| rs_rotation | long | 1682 | 0.107 | 1.31 | 0.48 |

## Per-regime expectancy by family

| family | side | mixed | risk_off | risk_on |
|---|---|---|---|---|
| mr_short_term | long | 0.000 (n=3516) | 0.001 (n=817) | 0.050 (n=76) |
| mr_short_term | short | 0.013 (n=5157) | 0.029 (n=132) | 0.132 (n=312) |
| multi_day_drop | long | 0.012 (n=2166) | -0.031 (n=329) | 0.086 (n=42) |
| overnight_hold | long | -0.006 (n=14813) | -0.029 (n=6488) | 0.012 (n=4122) |
| overnight_hold | short | -0.020 (n=19177) | 0.002 (n=734) | 0.000 (n=9989) |
| pead_wide | long | -0.062 (n=512) | 0.338 (n=49) | -0.002 (n=96) |
| pead_wide | short | 0.022 (n=284) | -2.086 (n=29) | 0.150 (n=32) |
| rs_rotation | long | 0.090 (n=1467) | 0.258 (n=151) | 0.149 (n=64) |

## Pooled rows vs shuffled baseline

| family | side | regime | n | exp_R | baseline exp_R | baseline n | excess | t_session | recovery | verdict | failed floors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| pead_wide | short | risk_on | 32 | 0.150 | 0.045 | 32 | 0.105 | 0.56 | 0.95 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb |
| mr_short_term | short | risk_on | 312 | 0.132 | 0.086 | 312 | 0.046 | -0.53 | 2.12 | WATCH | min_n, min_sessions, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| rs_rotation | long | risk_off | 151 | 0.258 | 0.238 | 151 | 0.020 | 1.13 | 3.09 | FAIL | min_n, min_sessions, min_decay, min_t_stat, min_t_stat_session, min_excess_r |
| pead_wide | short | mixed | 284 | 0.022 | 0.003 | 284 | 0.019 | 0.61 | 0.17 | FAIL | min_n, min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | short | risk_off | 734 | 0.002 | -0.003 | 733 | 0.005 | -1.62 | 0.11 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | short | risk_off | 132 | 0.029 | 0.025 | 132 | 0.004 | 0.88 | 0.81 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_rotation | long | mixed | 1467 | 0.090 | 0.086 | 1467 | 0.004 | 1.06 | 1.72 | WATCH | min_recovery, min_stability, min_t_stat, min_t_stat_session, min_excess_r |
| mr_short_term | short | mixed | 5157 | 0.013 | 0.009 | 5157 | 0.003 | -1.28 | 0.51 | WATCH | min_pf, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | long | risk_on | 4122 | 0.012 | 0.009 | 4119 | 0.003 | -0.16 | 1.33 | WATCH | min_exp_r, min_recovery, min_decay, min_t_stat_session, min_excess_r |
| mr_short_term | long | risk_off | 817 | 0.001 | -0.001 | 816 | 0.003 | -0.06 | 0.03 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | short | mixed | 19177 | -0.020 | -0.019 | 19128 | -0.001 | -4.60 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | long | risk_off | 6488 | -0.029 | -0.028 | 6448 | -0.001 | -1.15 | 0.00 | FAIL | min_pf, min_wr, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | long | mixed | 14813 | -0.006 | -0.005 | 14719 | -0.001 | -0.03 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| overnight_hold | short | risk_on | 9989 | 0.000 | 0.002 | 9961 | -0.002 | -1.92 | 0.01 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | long | mixed | 3516 | 0.000 | 0.004 | 3515 | -0.003 | 0.62 | 0.02 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| multi_day_drop | long | mixed | 2166 | 0.012 | 0.018 | 2166 | -0.006 | 0.59 | 0.36 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | long | mixed | 512 | -0.062 | -0.054 | 512 | -0.008 | -0.90 | 0.00 | FAIL | min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| rs_rotation | long | risk_on | 64 | 0.149 | 0.163 | 64 | -0.014 | 0.22 | 1.10 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| mr_short_term | long | risk_on | 76 | 0.050 | 0.074 | 77 | -0.024 | 0.78 | 0.67 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| multi_day_drop | long | risk_off | 329 | -0.031 | -0.006 | 329 | -0.024 | 0.06 | 0.00 | FAIL | min_n, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | long | risk_on | 96 | -0.002 | 0.025 | 96 | -0.027 | -0.84 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| multi_day_drop | long | risk_on | 42 | 0.086 | 0.150 | 42 | -0.064 | 0.86 | 1.30 | FAIL | min_n, min_sessions, min_recovery, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | long | risk_off | 49 | 0.338 | 0.453 | 49 | -0.116 | 0.97 | 1.70 | FAIL | min_n, min_sessions, min_recovery, min_stability, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |
| pead_wide | short | risk_off | 29 | -2.086 | -1.897 | 29 | -0.189 | -1.08 | 0.00 | FAIL | min_n, min_sessions, min_pf, min_exp_r, min_recovery, min_stability, min_decay, min_t_stat, min_t_stat_session, min_exp_r_lb, min_excess_r |

## Search accounting

| family | grid combos × sides | unconditional rows scored | PASS | WATCH |
|---|---|---|---|---|
| mr_short_term | 6 × 2 = 12 | 1002 | 0 | 0 |
| multi_day_drop | 4 × 1 = 4 | 496 | 0 | 0 |
| overnight_hold | 8 × 2 = 16 | 1005 | 0 | 58 |
| pead_wide | 9 × 2 = 18 | 641 | 0 | 0 |
| rs_rotation | 4 × 1 = 4 | 323 | 0 | 0 |
| **total** | 54 | 3467 | 0 | 58 |

5 families, 54 grid points × sides evaluated, 0 PASS

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
- Ten families times up to 27 parameter sets across hundreds of symbols will produce spurious PASS rows; the in-root shuffled baseline estimates how much of a pooled lane's expectancy is the regime's drift (see "Pooled rows vs shuffled baseline").
- Live indicators are computed on the in-progress bar; research uses closed bars. New lanes must use closed bars.

