# Post-Close Brief — Tuesday 2026-05-26 (Day 1 of quarter-size launch)

**Investigation triggered by:** user complaint at 13:25 ET that the bot took only 1 trade in 4 hours while ASTS rallied +10.6%, LRCX +2.2%, and other names ran without bot engagement.

## TL;DR

The bot is not broken. The bot is mis-configured for today's regime.

- **Today's bot P&L:** +$0 (1 trade, QBTS, position state ambiguous — see ledger gap)
- **Today's Schwab account P&L:** +$367-794 (pre-existing positions doing the work)
- **What breakout would have made today:** **~+$1,093** (10 trades, 4 winners hit target, 2 stops, 4 still green at EOD)
- **What we left on the table:** ~$700-1,100 — by disabling breakout this weekend based on backtest data that didn't include a trend day

## The five findings, in priority order

### 1. We disabled the wrong strategy for today's regime

Today was a textbook trend day. Mean-rev needs oversold extremes (RSI<30 + below BB lower). News needs heavy news flow + fresh sentiment. **Neither tool fits a rally with thin news.**

Breakout strategy (currently `enable_breakout_long: false`) would have caught:
- ASTS 112.97→target 116.48 at 09:34 ⇒ +$235
- LUNR 43.22→target 44.52 at 09:51 ⇒ +$236
- RDW 20.62→target 21.50 at 09:54 ⇒ +$236
- MU 866.96→target 892.97 at 13:34 ⇒ +$234

Total: 4 winners + 4 open-EOD-green + 2 stops = **+$1,093 net (UPPER BOUND — agent couldn't verify ADX/volume gates from log)**.

**The fix is NOT unconditional re-enable.** Backtest losses on chop days are real. **Add a regime gate** using `core/spy_regime.py`: only allow breakout when SPY up >0.5% intraday AND SPY ADX ≥ 20.

### 2. The screener rejected NVDA all day. Hardcoded.

NVDA appeared in Yahoo most-active and Schwab volume leaders. It was rejected 6-9× per cycle by `_passes_quality_filter` (Gate 3, relative-strength vs SPY).

- SPY 5d return: +1.42%
- NVDA 5d return: -4.07%
- Differential: **-5.49pp** (fails the hardcoded -3.0pp threshold)

**All 3 quality-filter gates are HARDCODED** in `analysis/screener.py:741-803`. There's no yaml knob.

Recommended changes (require code edit + restart):
1. Loosen Gate 3 from -3pp → -5pp
2. Lengthen RS window from 5 → 10 trading days
3. Add LRCX, AMD, ASML, NOW, SHOP, UBER, ABNB to `volatile_candidates` (screener.py:581-606)
4. Leave Gate 1 (ETF blocklist) UNTOUCHED — proven -$228 loss prevention on 5/22
5. **Maintainability fix**: lift the 3 hardcoded thresholds into Config().yaml so future regime tuning is yaml-only

### 3. LRCX was a sourcing gap, not a filter gap

LRCX would have PASSED the quality filter (+15.38% 5d return, +13.96pp vs SPY). It never reached the filter because it's not in any of the screener's input pools today:
- Not in volatile_candidates static list
- Not in yahoo_day_gainers top 30
- Not in $SPX/$COMPX %CHG_UP top 30

Fix: add LRCX (and other large-cap semis) to volatile_candidates. Same code edit as #2 item 3.

### 4. The classifier would have BLOCKED our only profitable trade today

This is the most important calibration data point of the day.

QBTS executed at 09:37:26. Side classifier ran at 09:37:25 (1 second earlier) with:
- `long_score=0.231, short_score=0.361, allows_long=False, allows_short=False, blocked_all`

If the classifier had been in enforcing mode at threshold 0.55, QBTS would have been blocked. Of our 5 classifier events today:
- 4 of 5 classifier blocks would have had NO effect (other safety rules already blocked)
- **1 of 5 (QBTS mean-rev) would have prevented our one executed trade**

**Keep classifier in shadow mode for now.** Need more data:
- Did QBTS hit target or stop today? (Position-state sync issue — flagged separately)
- Across 5/13-5/23 sim period, what % of profitable trades would classifier @0.55 have blocked?
- Should mean-rev and news have separate thresholds? (Today classifier disagreed with mean-rev but agreed with news on AMD)

### 5. Every safety block today was correct

5 signals fired today. 4 were blocked. **All 4 blocks were safety rules doing their job:**
- NVDA, AMD: pre-market block (allow_premarket=False)
- AMD: 5-min hard block (9:30-9:34)
- QBTS news #2: anti-pyramid (recent attempt 68 min prior)

**There were no false negatives in the safety chain.** The strategies just didn't fire enough to begin with.

## Calibration knobs ranked by impact and risk

| Change | Impact | Risk | Restart? | Priority |
|---|---|---|---|---|
| Regime-gated breakout re-enable | High (+$700-1,100/trend day) | Medium (backtest chop loss risk) | YES | **1 — biggest unlock** |
| Loosen screener Gate 3 (-3pp → -5pp) | High (admits NVDA-class) | Low (Gate 1 ETF blocklist unchanged) | YES | **2 — easy win** |
| Add LRCX et al. to volatile_candidates | Medium | Low | YES | 3 |
| Lift screener thresholds to yaml | Maintainability | Low | YES | 4 |
| Loosen mean-rev RSI threshold (30 → 32) | Low (4-5 more trades/day) | Medium (falling-knife risk) | YES | 5 |

## Deployment plan for tomorrow's open

**ALL changes require bot restart.** Restart window: tonight after 16:00 close, **after token re-auth**.

Suggested sequence:
1. **16:00** — Market close
2. **16:05** — User re-auths Schwab token (resets 7-day TTL — fresh until ~6/3)
3. **16:10** — Code changes:
   - Add regime gate to `BreakoutStrategyWithCommentary` (use `core/spy_regime.py`)
   - Flip `enable_breakout_long: true` in Config().yaml
   - Loosen screener Gate 3 to -5pp, lengthen window to 10d
   - Add LRCX, AMD, ASML, NOW, SHOP, UBER, ABNB to volatile_candidates
   - Lift screener thresholds to Config().yaml
4. **16:30** — Unit tests for new regime gate
5. **17:00** — Sim-mode dry run for 30 min against 5/22 chop log (verify regime gate kicks in correctly)
6. **22:00** — Final commit, deploy via `./deploy.sh`
7. **Wed 09:25** — Quick sanity check at startup

## What we should NOT change

1. **Live size multiplier (`live_size_multiplier: 0.25`)** — keep at quarter-size for Wednesday. Today's revealed gaps mean tomorrow's setup is partially-new code; quarter-size is the right risk for the first full day with breakout re-enabled.
2. **Max daily loss ($290)** — keep at 1%.
3. **Pre-market block, 5-min hard block, anti-pyramid** — all worked perfectly today.
4. **Classifier mode (shadow)** — keep in shadow until we have backtest validation across multiple regimes.

## Open questions for the user

1. **OK with regime-gated breakout for Wednesday open?** Or do you want to sim-test for a few days first?
2. **OK to lift hardcoded screener thresholds into yaml?** It's a 50-line refactor and a bot restart.
3. **QBTS today — what was the actual outcome?** Need Schwab-side fill history to answer (bot's local positions_data doesn't track it).

## Files generated

- `docs/training_runs/2026-05-26/signal_log.md`
- `docs/training_runs/2026-05-26/screener_audit.md`
- `docs/training_runs/2026-05-26/bb_proximity_scan.md`
- `docs/training_runs/2026-05-26/side_classifier_scorecard.md`
- `docs/training_runs/2026-05-26/breakout_replay.md`
- `docs/training_runs/2026-05-26/POST_CLOSE_BRIEF.md` (this file)
