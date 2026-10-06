# Adaptive RSI Turn — archive evaluation — 2026-10-06 10:32

Symbols with tick-rule minute bins: AAPL, AMD, AMZN, AVGO, COIN, DKNG, GOOGL, HOOD, INTC, META, MSFT, MU, NVDA, PLTR, ROKU, SMCI, SNAP, SOFI, TSLA, UBER. Window 2024-10-01 → 2025-03-12 (one bull regime).

## rsi_turn — no simulated trades

## Weight fit (Z, M, A; B at prior)

occurrences n = 8055, base rate P(+1R before stop) = 0.393, λ = 1.0, out-of-fold logloss = 0.6847

| component | fitted w | prior | final w |
|---|---|---|---|
| Z | 0.000 | 0.375 | 0.000 |
| M | 0.582 | 0.312 | 0.466 |
| A | 0.418 | 0.312 | 0.334 |
| B | — | 0.200 | 0.200 |

shrinkage η = 0.0 (0 = pure fit, 1 = prior) chosen on out-of-fold expected R = +0.113; θ: {'pooled': 50.0, 'mixed': 50.0, 'risk_off': 30.0, 'risk_on': 40.0}

By tape availability (occurrences with mf_* data vs without):

          size      mean
has_tape                
False     1134  0.387125
True      6921  0.394163

Base rate by month:

         size      mean
month                  
2024-10    78  0.000000
2024-11  1442  0.669903
2024-12  1716  0.449301
2025-01  1014  0.688363
2025-02  2233  0.252575
2025-03  1572  0.106870
