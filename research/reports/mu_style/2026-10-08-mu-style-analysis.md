# MU discretionary trades: what the data shows (2026-10-08)

Source: 12 external MU trades 2026-09-29..10-08 from trading_state.json (Schwab avg fill prices; entry/exit minute located by price match in yfinance 1m bars). Context CSV alongside.

## The trades
- 12 for 12 winners, +$11.9k; 100 sh typical, held ~1.5 h, both directions.
- Shorts: entered above VWAP (median +0.28%), upper third of day range (0.68), after a 30-60 min push up; median MAE 0.14%; exits below VWAP (-0.38%).
- Longs: mirror image below VWAP (-0.45%), lower part of range, after a dip; MAE 0.57%; exits above VWAP (+0.34%).
- Days traded: digestion days after a big prior move (median prior-day return -1.4%, small gap, many VWAP crosses). Not traded: trend days (Sep 22-24, Oct 1-2).

## Mechanical version, MU 1m bars 2026-09-14..10-08, 3 bps/side
Fade of VWAP extension (ext >= 0.3-0.8%, 30-min push >= 0.5-1%, stall 3-8 min, stop 0.5-1%, exit at VWAP or time): **0 of 48 cells positive**, median -10% summed. Tight-stop / target / trailing variants: **0 of 32 positive**. Restricted to the user's own traded days: still -2.7% over 18 trades while the user made +7% on those days. Only an ex-post "low trend day" filter is positive (7 trades) and it is not knowable in advance.
Fit pipeline agrees over a longer history: vwap_reversion on MU pf 0.66 (long, n=203) / 0.72 (short, n=201); mean_rev_intraday long pf 0.94.

## Conclusion
The money came from timing inside the day (precise entries at the stall, exits held through VWAP, and today's short held into a 3.4% break), not from the VWAP-fade rule itself, and not from a day filter visible in bars. 12 trades is also a small sample. A minute-bar encoding of this style loses; the discretionary component needs either finer data (tape) or the trader's own stated trigger.

## Encoded as family `prior_high_rsi` (2026-10-08)
`strategies/families/prior_high_rsi.py`: short on a failed retest (within near_pct) of the session's prior swing high — the extreme of the 24 five-minute bars ending 3 bars earlier, so a high set on the previous bar is a continuation, not a level — with RSI(14, 5-min) >= rsi_short; long mirror at a prior low with RSI <= 100 - rsi_short; stop stop_pct beyond the level, target target_r x risk, 24-bar time stop, entries until 14:30. Grid 2x2x2x3 = 24. Level computation is vectorised (~9 ms on a 100k-bar frame).
Sanity run through `indicator_frame` on MU 5-min bars 2026-09-14..10-08 (bracket sim, 3 bps/side), params (near, rsi, stop, target_r): (0.3,60,0.3,1.5) 16 entries, net +0.05%, win 43%, exits {'stop': 7, 'target': 5, 'time': 4}; (0.6,60,0.6,1.5) 21 entries, net -0.21%, win 42%, exits {'time': 15, 'stop': 4, 'target': 2}; (0.6,65,0.6,2.0) 10 entries, net -1.30%, win 40%, exits {'time': 7, 'stop': 3}. Several fire within minutes of the user's own entries (09-29 10:15 short, 09-30 10:50 short, 10-08 12:25 short). Enabled as a shadow fit lane with MU pinned and ledgering before the nightly fit has rows; no live entries. Decision after two weeks of ledger vs. the user's fills.
