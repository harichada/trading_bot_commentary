# Plan — Symbol Side Classifier

**Status**: DRAFT for operator review (2026-05-13).
**Trigger**: when `enable_mean_rev_short=true` was flipped on 2026-05-11, only
shorts fired (5 entries, all into momentum continuation, -$665 unrealized).
The next day (2026-05-12) all longs were also bad — falling-knife buys on stale
news + oversold mean-reversion in active downtrends, -$2.4K phantom-loss
display. Naive symmetric enable of both sides produces asymmetric pain.

This doc proposes a **side-selection classifier** that decides, per symbol per
session, which sides are even *allowed* to enter — before any strategy emits a
signal. No code is written yet; this is for review and redirect.

---

## 1. Problem statement

The bot has multiple strategies (news, mean-reversion, breakout in
`strategies/builtin.py`) that each emit BUY or SELL signals independently.
None of them looks at the **broader regime of the symbol they're firing on**:

- Mean-reversion LONG fires `RSI<30 AND close<BB_lower`. That's true at the
  *bottom of every downtrend in progress*. Without a trend filter, it
  catches falling knives.
- Mean-reversion SHORT fires `RSI>70 AND close>BB_upper`. That's true at
  every overbought *uptrend pullback peak*. Without a trend filter, it
  shorts strength in bull moves.
- News BUY has four filters (falling-knife, rising-knife, late-entry,
  price-direction). News SHORT branch has none of those.

Today the bot operates on a single watchlist — the screener's top movers.
That universe is biased toward **whatever's moving today**. On a strong tape,
that biases overbought; on a weak tape, oversold. Either bias dominates the
strategies that don't account for it.

Yesterday's mean-rev shorts (RSI 92, 78, 75, 72, 74) and today's mean-rev
longs (RSI 24-26, ORCL/CRWV in multi-hour downtrends) were both this exact
failure mode, on opposite ends.

## 2. Why naive enabling fails

Three observed mechanisms, each independently sufficient:

| # | Mechanism | Evidence |
|---|---|---|
| 1 | Universe skew — screener's "active stocks" are momentum names; mean-rev fires at *whichever* extreme the momentum has reached | 2026-05-11: IONQ RSI 92, NVTS 91, NVDA 83 all flagged short; no symmetric oversold names |
| 2 | Strategy logic is *price-extreme-based*, not *reversal-confirmed* — RSI/BB are computed from the same price action that's still trending | 2026-05-12: CRWV declined 63 min before mean-rev fired BUY at RSI 24; price kept declining after entry |
| 3 | Each strategy reimplements its own filter chain — fixing one (news late-entry-guard, 2026-05-08) doesn't fix the other (mean-rev has zero overlap) | Today's analysis: news refused ORCL 8× with `price_direction_disagrees_buy`; mean-rev bought it on the 9th cycle |

A single side-selection layer cures (1) by refusing signals against the
prevailing trend. It mitigates (2) and (3) by being the **one place** that
asks "is this symbol set up for this side *at all today*?"

## 3. Goal — what the classifier returns

```
SymbolSideDecision:
    allowed_sides: frozenset[Side]       # subset of {LONG, SHORT}
    long_score: float                    # 0.0 (no long edge) .. 1.0 (strong long bias)
    short_score: float                   # 0.0 (no short edge) .. 1.0 (strong short bias)
    primary_reason: str                  # one line for commentary / audit
    feature_dump: dict                   # full input vector for post-mortem
```

Four observable outcomes per symbol:

| `allowed_sides` | Meaning | Example |
|---|---|---|
| `{LONG}` | Symbol in uptrend; only buy-the-dip signals proceed | NVDA in a clean uptrend day |
| `{SHORT}` | Symbol in downtrend; only short-the-bounce signals proceed | A name making lower lows on rising volume |
| `{LONG, SHORT}` | Range-bound, valid either way | A ticker chopping in a 2-day range |
| `∅` (empty) | Don't trade this symbol today either way | Earnings tomorrow / parabolic vertical / no liquidity |

The classifier is **explicit about doing nothing**. `∅` is a normal outcome,
not an error.

## 4. Feature set

Per symbol, computed once and cached for the session (or 30 min TTL,
whichever is shorter):

### 4.1 Trend (the load-bearing features)

| Feature | Computation | Side it argues for |
|---|---|---|
| `daily_trend_50` | sign(close − SMA_50_daily) | `+` ⇒ LONG bias, `-` ⇒ SHORT bias |
| `daily_trend_20` | sign(close − SMA_20_daily) | same |
| `sma_alignment` | `SMA_20_daily > SMA_50_daily` (uptrend if true) | both above ⇒ LONG, both below ⇒ SHORT |
| `intraday_trend` | regression slope of 5-min closes over today's session | confirms or contradicts daily |
| `lower_lows` | count of consecutive lower-low daily bars in last 5 | ≥ 3 ⇒ SHORT bias; never LONG |
| `higher_highs` | count of consecutive higher-high daily bars in last 5 | ≥ 3 ⇒ LONG bias; never SHORT |

**Hard gate**: if `daily_trend_50` and `daily_trend_20` disagree, the symbol
is *transitioning* — neither side is allowed without supporting evidence.

### 4.2 Position within the range (for entry quality)

| Feature | Computation | Use |
|---|---|---|
| `pct_from_sma20` | `(close − SMA_20) / SMA_20` | LONG edge best at small negative; SHORT edge best at small positive |
| `pct_from_52w_high` | `(close − high_52w) / high_52w` | LONG more reasonable away from extremes |
| `pct_from_52w_low` | `(close − low_52w) / low_52w` | symmetric |
| `daily_atr_pct` | `ATR_14_daily / close` | sets risk distance; very high ⇒ harder either side |

### 4.3 Volume / participation

| Feature | Computation | Argument |
|---|---|---|
| `vol_ratio_today` | `today_volume / avg_volume_20d` | confirms conviction |
| `up_vs_down_volume_5d` | sum(volume on up days) / sum(volume on down days), last 5 | `>1` ⇒ accumulation ⇒ LONG; `<1` ⇒ distribution ⇒ SHORT |

### 4.4 News flow

| Feature | Source | Argument |
|---|---|---|
| `news_sentiment_7d` | aggregated daily sentiment, last 7 days | strongly positive ⇒ block SHORT; strongly negative ⇒ block LONG |
| `news_fresh_count_today` | articles in last 4h | fewer than 2 ⇒ no news edge; doesn't gate either side, just removes news-induced bias |
| `news_recency_min` | minutes since latest article | feeds the existing news strategy's veto chain |

### 4.5 Relative strength

| Feature | Computation | Argument |
|---|---|---|
| `rs_vs_spy_5d` | symbol return − SPY return, last 5 sessions | strong outperformance ⇒ LONG bias; strong underperformance ⇒ SHORT bias |
| `rs_vs_sector_5d` | same against the sector ETF | same |

### 4.6 Event blackouts (hard gates → `∅`)

- Earnings within ±2 trading days
- Ex-dividend within ±1 day (long bias affected)
- Known FDA/PDUFA dates (for biotech subset)
- Halt history today (do not trade)

### 4.7 Borrowability (hard gate for SHORT)

- Hard-to-borrow flag from broker (if available) ⇒ remove SHORT from
  `allowed_sides` regardless of score
- Recent short squeeze pattern (rapid gap-up + low float) ⇒ same

## 5. Decision logic

Two-stage: **hard gates** then **soft scoring**.

### 5.1 Hard gates (any one fires ⇒ result is `∅`)

- Earnings within ±2 days
- Halt history today
- `daily_atr_pct > 8%` (regime too volatile for systematic side selection)
- Price below `MIN_PRICE` (already exists; surfaces here for symmetry)
- Daily volume < 500k (illiquidity)

### 5.2 Hard side-only gates

- Hard-to-borrow ⇒ remove SHORT
- `lower_lows ≥ 3` ⇒ remove LONG (no buy-the-dip in confirmed downtrend)
- `higher_highs ≥ 3` ⇒ remove SHORT (no short the strength in confirmed uptrend)
- `news_sentiment_7d > +0.6` ⇒ remove SHORT (no shorting strongly bullish news flow)
- `news_sentiment_7d < −0.6` ⇒ remove LONG (no longing strongly bearish news flow)

### 5.3 Soft scoring → `long_score`, `short_score`

Linear combination of features, each contribution in `[-1, +1]` toward the
appropriate side's score, then `sigmoid` to `[0, 1]`. Weights start as a
simple intercept-free LR with operator-tuned coefficients:

```
long_score  = sigmoid(  +0.30 * sma_alignment_up
                        +0.20 * up_vs_down_volume_5d_normalized
                        +0.15 * rs_vs_spy_5d_normalized
                        +0.15 * intraday_trend_normalized
                        +0.10 * news_sentiment_7d_clamped_positive
                        +0.10 * pct_from_sma20_negative_small)
short_score = sigmoid(  +0.30 * sma_alignment_down
                        +0.20 * down_vs_up_volume_5d_normalized
                        +0.15 * rs_vs_spy_5d_inverse_normalized
                        +0.15 * intraday_trend_normalized_inverse
                        +0.10 * news_sentiment_7d_clamped_negative
                        +0.10 * pct_from_sma20_positive_small)
```

Initial threshold: a side is `allowed` if its score ≥ 0.55 (modest edge).
Both sides above 0.55 is the `{LONG, SHORT}` (range-bound) case.

The weights and threshold are tunable from `Config().yaml`. They are NOT
sacred — operator decides post-backtest.

## 6. Integration into the pipeline

```
signal_router.process(signal):
    ⋮
    classifier_decision = symbol_classifier.classify(signal.symbol)   # cached
    if signal.side not in classifier_decision.allowed_sides:
        audit("symbol_classifier", signal.symbol, "skip",
              reason=f"side_not_allowed_{signal.side}",
              long_score=...,
              short_score=...,
              primary_reason=classifier_decision.primary_reason)
        return
    # existing health gate, price filter, correlation guard, etc.
    ⋮
```

Placement: **after** the strategy emits a signal, **before** any of the
existing filters fire. Rationale: strategies stay simple; the classifier
is the one place that knows "is this symbol set up for this side." Every
strategy automatically inherits the gate.

The audit log line names the classifier explicitly so future operator
reviews can grep `engine_decision component=symbol_classifier` and see
every blocked entry with full scoring context.

## 7. Caching, freshness, hysteresis

- **Cache TTL**: 30 minutes. The features that drive the decision (daily
  trend, multi-day RS, 7-day sentiment) move on a multi-day timescale.
  Intraday flips would be whipsaw.
- **Cold-start**: at engine startup, classify every position currently
  held (so the operator sees the per-symbol classification on the
  dashboard immediately) plus the current watchlist.
- **Hysteresis**: a symbol that was `{LONG}` yesterday and is `{SHORT}`
  today requires an explicit "regime flip" log line at WARNING level so
  the operator can sanity-check it. Don't silently invert sides.

## 8. Failure modes and mitigations

| Failure | Mitigation |
|---|---|
| Daily data fetch fails (Schwab API hiccup) | If `< 50` daily bars cached for the symbol, hard gate to `∅`. Don't pretend to classify with insufficient data. |
| News aggregator stale | Drop the news features from scoring; don't gate. Soft scores still compute on technicals alone. |
| All features agree but symbol is in a manipulation event (pump, short squeeze in progress) | Volume-spike detector adds a `vol_ratio_today > 5` rule that forces `∅` regardless of score. |
| Classifier itself crashes during signal eval | Fail-closed: signal is dropped, WARNING logged, supervisor restart triggered. Prefer skipping a trade to entering blind. |
| Operator wants to override a classification temporarily | `Config().yaml` lists `manual_overrides:` with `{LONG_ONLY:[…], SHORT_ONLY:[…], FORBIDDEN:[…]}`. Operator additions take precedence over the classifier. |

## 9. Validation before live

The classifier must pass two gates before it's allowed to *block* live entries:

1. **Backtest replay (mandatory)** on the last 90 sessions of the existing
   watchlist universe. Report: per-symbol-per-day decisions; outcomes of
   entries the classifier would have allowed vs blocked; net P&L delta vs
   baseline. Acceptance: blocked-entry P&L is net **worse** than baseline
   (i.e., the classifier is filtering the *bad* trades, not the *good*
   ones).
2. **Shadow mode in live (mandatory, ≥ 5 sessions)**. Classifier runs and
   logs every decision, but the gate is bypassed. Operator reviews the
   shadow-decision log next morning; if the decisions agree with what the
   operator would have wanted, the gate is activated.

No live gating until both gates pass. This is the same posture as
`v-news-verifier-advisory-2026-05-08`.

## 10. Phased rollout

| Phase | What's enabled | How long |
|---|---|---|
| 0 — design | This doc reviewed + redirected | now |
| 1 — implementation | Classifier + tests + audit logging, gate **off** | 1-2 dev days |
| 2 — backtest | Replay 90 sessions, produce report | 1 day |
| 3 — shadow live | Classifier decisions logged, no gating | ≥ 5 sessions |
| 4 — partial gate | Hard gates only (earnings, halts, HTB, daily_atr) | ≥ 5 sessions |
| 5 — full gate | All hard + soft (`allowed_sides`) gating | indefinite |

Stop the rollout at any phase if a gate behaves unexpectedly. Each
transition is operator-approved, not automatic.

## 11. Open questions for operator review

1. **Daily-bar data source**. Currently the bot fetches 5-min bars via
   `SchwabDataProvider.get_market_data(period_type='day', period=1)`.
   For daily SMA50/SMA20 we need 60 trading days of daily bars. Reuse
   the same Schwab API with `frequency_type='daily'`? Cache per symbol?
   The existing screener doesn't pull this; this is new I/O.
2. **News history depth**. The current news system caches per-symbol but
   I don't think it goes back 7 days. We may need a small per-symbol
   sentiment-history table.
3. **Hard-to-borrow data**. Does the Schwab API expose this on the
   account/quote endpoint, or do we need to maintain a manual list?
4. **Sector mapping**. Per-symbol sector for `rs_vs_sector_5d` —
   adopt a small embedded map (S&P 500 + popular ADRs) or hit an
   external lookup?
5. **Threshold of 0.55**. Conservative-but-arbitrary; do you want it
   higher (fewer trades, higher conviction) or lower (more trades)? The
   backtest in phase 2 will inform this; ok to start at 0.55 and let
   the data redirect.
6. **Manual overrides**. Operator-supplied symbol lists in
   `Config().yaml`: useful, or noise? My instinct says keep them — there
   will be days where you *know* a name is shortable that the classifier
   misses, and a manual override is faster than tuning weights.
7. **Where does this live structurally?** Proposed `core/symbol_classifier.py`.
   It will be a producer task in the parallel-loops architecture once
   that's complete — a `classifier_loop` that refreshes decisions every
   30 min and writes to a shared store. For now (single monolithic
   analysis_loop), inline cache + lazy eval on first use per symbol.

## 12. What this is NOT

- Not an ML model in v1. Initial weights are operator-judged linear
  coefficients on hand-engineered features. ML comes later if backtest
  shows the linear model leaves edge on the table.
- Not a trade-quality scorer. It does not say "this trade is good"; it
  says "this *side* is allowed on this symbol today." Trade quality is
  what the strategies + existing filters already evaluate.
- Not a market-regime detector. SPY going up or down is *one feature*
  (relative strength input), not the whole decision.
- Not a replacement for the news strategy's existing four filters. Those
  stay. The classifier is upstream of them.

---

## Summary of what you're approving (if you approve)

1. A new `core/symbol_classifier.py` module with the feature set in §4
   and the decision logic in §5.
2. A new audit-log component (`engine_decision component=symbol_classifier`)
   that records every allowed/blocked decision with full scoring.
3. Integration at the top of `signal_router.process()` so every strategy
   feeds through it.
4. A 5-phase rollout that ends with the gate active only after backtest
   and shadow validation.
5. Operator-tunable weights and threshold in `Config().yaml`.

The whole point is: **next time you enable shorts, only shortable symbols
get shorted, and only longable symbols get longed, and most stocks get
neither side allowed** (because most stocks on any given day aren't a
clean setup either way).

Tell me what to redirect.
