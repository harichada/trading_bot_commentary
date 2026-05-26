# Bollinger Band Proximity Scan — 2026-05-26

Mean-reversion `no_setup` rejection analysis. Read-only snapshot at 13:30 ET.

## Headline

Of 1,537 mean-rev `no_setup` rejections today:
- **184 (12.0%)** within **0.5%** of BB lower band (very close to setup)
- **405 (26.4%)** within **1%** of BB lower
- **746 (48.5%)** within **2%** of BB lower
- **1260 (82.0%)** within **5%** of BB lower

68 unique symbols evaluated by mean-rev today.

## Top 25 near-miss candidates today (closest to BB lower, sorted by proximity)

| Distance | Symbol | RSI | Close | BB lower | Note |
|---|---|---|---|---|---|
| -1.63% | BRAI | 41.2 | 13.90 | 14.13 | RSI too high |
| -1.05% | APLD | 33.6 | 47.11 | 47.61 | RSI marginal |
| -0.94% | P | 34.1 | 88.07 | 88.91 | RSI marginal |
| -0.89% | QBTS | 34.2 | 28.37 | 28.62 | (we traded earlier at 24.88) |
| -0.87% | OKLO | 41.6 | 70.10 | 70.72 | RSI too high |
| -0.79% | INFQ | 31.2 | 16.26 | 16.39 | very close, RSI just above 30 |
| -0.70% | QCOM | 33.6 | 241.03 | 242.73 | RSI marginal |
| -0.65% | IONQ | 31.9 | 62.81 | 63.22 | near-setup |
| -0.64% | QUBT | 32.2 | 12.05 | 12.13 | near-setup |
| -0.63% | FLY | 43.4 | 56.85 | 57.21 | RSI too high |
| -0.62% | BB | 30.1 | 8.07 | 8.12 | **RSI 30.1 — sat exactly on threshold** |
| -0.42% | RKLB | 35.5 | 139.65 | 140.24 | |
| -0.38% | ASTS | 44.7 | 122.03 | 122.5 | RSI too high |
| -0.11% | ASTS | 48.1 | 122.88 | 123.02 | RSI too high (and rallying) |
| -0.10% | KEEL | 36.8 | 5.02 | 5.02 | RSI too high |
| -0.09% | JOBY | 46.0 | 11.09 | 11.10 | RSI too high |

## Pattern observations

1. **RSI threshold appears to be the binding constraint, not BB proximity.** Many near-BB candidates failed because RSI was in 30-50 range, not the oversold <30 zone the strategy targets.

2. **QBTS RSI=34.2 at -0.89% from BB lower** — got close to firing a SECOND entry later in the day. Our earlier QBTS trade entered at RSI=24.88 with distance_pct=5.76 (well below band).

3. **BB at RSI=30.1 was the closest "almost triggered" event.** Distance from BB lower: -0.62%. If RSI threshold were 31 instead of 30, this would have fired. Single-point sensitivity.

4. **ASTS appears in the table but didn't fire** because RSI was 44-48 (not oversold) AND it was rallying (price rebounding from the band on a bounce). Mean-rev correctly didn't engage — ASTS wasn't a mean-rev play, it was a momentum play (which we have no strategy for).

## Calibration implications

**If we lowered RSI threshold from 30 → 32** (RSI-only relaxation):
- Would have captured: BB (8.07), INFQ (16.26), IONQ (62.81), QUBT (12.05), QBTS (28.37 — 2nd attempt, blocked by anti-pyramid anyway)
- Risk: falling-knife exposure increases. Many of these names also showed `falling_knife` rejection elsewhere in the log (need separate analysis to confirm).

**If we widened BB proximity from "below band" → "within 0.5% of band"** (BB-only relaxation):
- Adds 184 events as potential setups
- Most still gated out by RSI threshold

**If we relaxed both** (RSI <32 AND within 1% of BB lower):
- Could see 5-15 additional setups per day in normal conditions
- This is the dial that controls trade frequency vs. quality tradeoff

## Recommended action for tonight's review

1. **Cross-tab RSI-near-miss × falling_knife rejection** to see what fraction of the 184 near-misses would ALSO fail falling-knife (and thus relaxing RSI alone wouldn't help).
2. **Run the BB at RSI=30.1, close=8.07 trade hypothetically** — what would the outcome have been? Single-trade replay.
3. **Decide if RSI=32 + BB-buffer expansion is worth running in sim** for a few days before live.
