# rsi_peak_fade — the user's short rule as a family (2026-10-15)

Replaces `prior_high_rsi` (2026-10-08). The user's rule, stated 2026-10-15: "shorting is done when the RSI is max and
about to fall". Their 8 MU shorts (09-29..10-08) sit 1-3 five-minute bars after the 5-min RSI peak (6 of 8), on the first
stall, with 1-min RSI having peaked 65-86 in the prior 10 minutes, price within ~0.2-0.5% of the bounce high, median
adverse excursion 0.14%. MU was 1-12% ABOVE its 20-day EMA on every one of those days: the "weak on the daily" context I
first assumed was wrong, so the context is an optional *extension* filter (`ext_min`), not a weakness filter.

## Rule (short; long mirrors against 100 - peak_min)
1. 5-min RSI(14) peak >= `peak_min` within the last 3 bars, the highest RSI of the 30 min before it.
2. No print at/above the peak since; trigger-bar RSI >= 3 points below the peak and still >= 50 (no chasing).
3. Trigger-bar close within 0.4% of the highest high from the peak bar through the trigger bar; peak bar in-session.
4. `ext_min` > 0: lagged daily close >= ext_min % above the 20-day EMA.
5. Entry next open; stop = max(bounce high + 0.05%, entry + 0.3%); target = target_r x risk; 24-bar time stop;
   no trigger after 14:50 ET; one per session per side.
Grid: peak_min {55,60,65} x ext_min {0,3} x target_r {1,1.5,2} = 18.

## Sanity on MU, 20 sessions 2026-09-18..10-15 (5-min bars from yfinance 1-min, bracket sim, no costs)
| ext_min | peak_min | n (shorts) | sum R @1.5 | win | shorts within 45 min of a user short |
|---|---|---|---|---|---|
| 0 | 55 | 29 (15) | +3.3 | 48% | 6 |
| 0 | 60 | 23 (9) | +0.4 | 43% | 3 |
| 0 | 65 | 14 (6) | +3.5 | 50% | 2 |
| 3 | 55 | 12 (10) | +2.1 | 50% | 4 |
| 3 | 60 | 8 (6) | +1.1 | 50% | 3 |
| 3 | 65 | 5 (3) | 0.0 | 40% | 2 |
The peak_min=55 variant fires within 5 minutes of the user's 10-05 12:54, 10-06 11:02 and 10-08 12:33 shorts (all
three won in the sim). Twenty sessions of one symbol prove nothing about expectancy; the point is that the trigger now
lands where the user's fills are. The nightly hunt grades it across the universe from 2026-10-15 22:30; the live fit lane
ledgers it in shadow (`data/shadow_fit_rsi_peak_fade.ndjson`) from the first restart after the 10-15 close.

Also 2026-10-15: `day_trade_short_live_entries_enabled` set false (live via PUT /api/settings/trading at 11:04 ET) —
that lane sold the low (continuation_down, RSI 30-50) and was 0/5 live, -$145; its max_concurrent=2 was never enforced.
