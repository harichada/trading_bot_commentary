# Side Classifier Scorecard — 2026-05-26

Shadow-mode evaluation. Read-only snapshot at 13:30 ET.

## Confirmation: classifier is in shadow mode, not gating

```
side_classifier: shadow mode ENABLED (model_path=models/classifier/ffn_watchlist_v1.pt)
— decisions logged, no gating applied.
```

Confirmed: every `action=shadow` event today was logging-only. NONE blocked a trade. The mid-day "no trades" perception was NOT caused by the classifier.

## Today's 5 classifier events (all shadow mode)

| Time (ET) | Symbol | Strategy signal | Rule says | Trained says | long_score | short_score | If ENFORCING @0.55, would have... |
|---|---|---|---|---|---|---|---|
| 09:20:35 | NVDA | news BUY | long+short allowed | blocked_all | 0.400 | 0.206 | **BLOCKED** (also blocked by pre-market) |
| 09:26:45 | AMD | news BUY | long allowed | blocked_all | 0.385 | 0.248 | **BLOCKED** (also blocked by pre-market) |
| 09:30:44 | AMD | news BUY | long_only score=0.77 | **agrees** | 0.424 | 0.249 | **BLOCKED** (also blocked by 5-min hard block) |
| **09:37:25** | **QBTS** | **mean-rev BUY** | **short_only allowed** | **blocked_all** | **0.231** | **0.361** | **BLOCKED — would have prevented our ONE executed trade** |
| 10:45:32 | QBTS | news BUY | short_only allowed | blocked_all | 0.297 | 0.246 | **BLOCKED** (also blocked by anti-pyramid) |

## The headline insight: classifier disagrees with strategy on QBTS

**QBTS at 09:37:25** is the critical data point.

- Mean-rev strategy fired: RSI=24.88, distance_pct=5.76 below BB lower → "oversold bounce BUY"
- Side classifier (rule-based): "short_only" — i.e., this should be a SHORT trade, not long
- Side classifier (trained model): `blocked_all` (long_score 0.23, short_score 0.36) — neither side has edge

**The strategy thinks QBTS is reverting up. The classifier thinks QBTS is still falling.**

We executed the strategy's view. Whether QBTS hit target ($29.03) or stop ($26.52) determines who was right. **Need post-close fill data to settle this empirically.**

## Would enforcing have helped or hurt today?

| Event | Enforcing classifier blocks? | Was already blocked by other safety? | Net effect |
|---|---|---|---|
| NVDA news | YES | YES (pre-market) | No change |
| AMD news #1 | YES | YES (pre-market) | No change |
| AMD news #2 | NO (agrees with strategy) | YES (5-min hard block) | No change |
| **QBTS mean-rev (executed)** | **YES** | NO (passed all other gates) | **Trade not placed → unknown P&L impact** |
| QBTS news #2 | YES | YES (anti-pyramid) | No change |

**Net: enforcing the classifier today would have changed exactly one outcome — block QBTS mean-rev.** Every other classifier block was already covered by another safety rule.

## Calibration questions for tonight

1. **Did QBTS hit target or stop?** If target: classifier was wrong, threshold 0.55 is too strict. If stop: classifier was right, raising threshold is justified.
2. **Backtest the classifier**: across the 5/13–5/23 sim period, what % of trades would have been blocked by classifier @0.55? What was the avg P&L of blocked-vs-allowed?
3. **Decision threshold sensitivity**: at threshold 0.50, 0.55, 0.60 — what's the precision/recall on profitable trades?
4. **Per-strategy threshold**: should mean-rev (which classifier disagreed with on QBTS) have a different threshold than news (which classifier agreed with on AMD)?

## Position state oddity (separate issue, flagged)

QBTS order placed at 09:37:27 today but NOT in `trading_state.json:positions_data` (only COIN, EOSE, HQGE, RIVN, PINS are tracked).

Possible explanations:
- OCO bracket order is broker-side (Schwab tracks, bot doesn't sync local state)
- `manual_close_only=True` config means bot delegates entry+exit tracking to Schwab
- Position-state sync bug (less likely)

This warrants a separate investigation, NOT urgent.
