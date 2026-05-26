# Signal Log — 2026-05-26 (Tuesday, Day 1 of quarter-size launch)

Read-only post-mortem snapshot taken 13:24 ET while bot is still live.
P&L at snapshot: +$581 (peak $794 at 12:11 ET).

## All strategy signals fired today

| # | Time (ET) | Symbol | Strategy | Reason | Entry | Stop | Target | R | Outcome |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 09:20:35 | NVDA | news | high_impact_news (sentiment=0.51, fresh=7, age=30.8min) | 217.70 | 214.43 | 224.23 | 1.0R/2.0R | blocked: pre-market |
| 2 | 09:26:45 | AMD | news | high_impact_news (sentiment=0.28, fresh=3, age=37.7min) | 484.70 | 477.43 | 499.24 | 1.0R/2.0R | blocked: pre-market |
| 3 | 09:30:44 | AMD | news | high_impact_news (sentiment=0.33, fresh=3, age=41.7min) | 485.04 | 477.77 | 499.60 | 1.0R/2.0R | blocked: 5-min hard block |
| 4 | 09:37:25 | QBTS | mean_rev | oversold_bounce (RSI=24.88, dist=5.76%) | 27.45 | 26.52 | 29.03 | 1.0R/1.7R | **EXECUTED qty=31** |
| 5 | 10:45:32 | QBTS | news | high_impact_news (sentiment=0.74, fresh=2, age=62min) | 27.09 | 26.57 | 28.14 | 1.0R/2.0R | blocked: anti_pyramid (age=68min) |

## Block-reason breakdown

- **3 of 5 (60%)** blocked by safety rules — pre-market, 5-min hard block, anti-pyramid. All correct.
- **1 of 5 (20%)** executed — QBTS mean-rev.
- **1 of 5 (20%)** would be candidate for signal-persistence retry (AMD pre-market at 09:26 could re-evaluate at 09:35+).

## What-if recovery analysis (pending Agent 1 results)

If NVDA and AMD signals had been allowed to re-evaluate at 09:35+ (post-hard-block):
- Need: actual NVDA, AMD intraday closes for 09:35, 12:00, 14:00, 16:00
- Need: would the signal still have fired (sentiment, freshness, price action still favorable)?
- Estimate: at quarter-size with $290 daily-loss cap, max 1-2 simultaneous live entries.

**Deferred until Agent 1 (breakout replay) provides intraday price data for NVDA/AMD.**

## Side classifier observations (shadow mode, not gating)

5 side_classifier events logged today. All `action=shadow` (logging only, classifier is NOT enforcing yet).
Score distribution:
- NVDA long_score=0.4, short_score=0.21 → trained model said `blocked_all`
- AMD long_score=0.42, short_score=0.25 → trained model said `agrees with rule` (rule_allowed long_only, score 0.77)
- QBTS long_score=0.30, short_score=0.25 → `blocked_all`

Calibration question for tonight: if classifier had been in ENFORCING mode at threshold 0.55, what would have changed today?
- NVDA: classifier blocks (already blocked by market_hours, no change)
- AMD #1: classifier passes long (already blocked by market_hours, no change)
- AMD #2: classifier passes long (already blocked by 5-min, no change)
- QBTS mean-rev: NOT in this log set (need separate trace) — was it shadowed too? **Critical question: did classifier shadow-block our one successful trade?**
- QBTS news #2: classifier blocks (also blocked by anti-pyramid, no change)

**Today's classifier vs. safety-rule overlap: 100%.** Every classifier shadow-block was also blocked by something else.

## Open positions context (NOT placed today, pre-existing)

| Symbol | Qty | Entry | Side | Note |
|---|---|---|---|---|
| COIN | 42 | 197.83 | long | entry_time 2026-05-26T08:19 (pre-bot-start) |
| EOSE | 391 | 9.22 | long | same |
| HQGE | 5100 | 0.019 | long | same |
| RIVN | 644 | 14.99 | long | same |
| PINS | 292 | 23.66 | long | same |

These are NOT bot trades today. The +$581 P&L is mostly from these 5 positions moving.

## Strategy coverage gap (the headline problem)

Today's market: bullish rally with thin news.
- Mean-rev needs RSI<30 + below BB lower → today's market has RSIs 70-90 in most names → 1 setup all day (QBTS)
- News needs ≥5 fresh articles + strong sentiment → thin news day → most symbols 1-4 articles
- Breakout disabled this weekend → would have engaged this regime
- No trend/momentum strategy → ASTS +10.6% watched but no tool to engage

**Result: 5 signals in a full trading day, 1 executed.**
