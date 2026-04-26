# Trading Bot Upgrade — Expert Prompt Pack

> **Role framing for every prompt below.** Preface each with:
> *"You are a senior quantitative strategist with 15+ years at a systematic hedge fund. You have shipped production alpha for equities and derivatives, read López de Prado's *Advances in Financial Machine Learning* as gospel, and refuse to ship anything that hasn't survived purged walk-forward validation with realistic costs. You prefer **fewer, higher-conviction trades** over activity. You never fit in-sample. Cite the methodology behind every choice."*

---

## Diagnosis (why this pack exists)

| Area | Current state | Impact |
|---|---|---|
| Strategies | 4 built-ins, PF 0.78–0.83 in backtests | **Negative edge** |
| Signal combination | Parallel OR gate, no meta-label | False positives dominate |
| Regime conditioning | Detector exists, not wired into sizing/gating | Wrong strategy at wrong time |
| ML pipeline | VotingClassifier, no calibration, no SHAP, no drift detection | Probabilities unreliable |
| Risk sizing | Fixed-frac + Kelly, no vol targeting, no correlation cap | Sector concentration risk |
| Execution | Market / single-limit bracket only | Slippage leaks alpha |
| Commentary | Template-based, `use_llm_analysis: false` | Flat, non-adaptive |
| Backtest | No walk-forward rolling retrain, no Monte Carlo | Optimistic |

**Thesis:** The fastest path to profitability is **not more strategies** — it's (a) a meta-labeling layer that learns *when to trust* the existing signals, (b) regime-gated position sizing with volatility targeting, (c) cost-aware objectives, and (d) an LLM-driven commentary/decision brain that produces calibrated conviction scores from heterogeneous evidence.

---

# TIER 1 — Profitability Foundation (do these first)

### P1. Meta-labeling filter on every strategy (López de Prado Ch. 3)

> Objective: Turn the existing primary strategies from *PF 0.8* into *PF 1.4+* by learning a **secondary classifier** that predicts whether a primary signal will reach 1R before hitting stop, conditional on regime, microstructure, and recent performance.
>
> Context: Current primaries in `strategies/builtin.py` (Momentum, Breakout, MeanReversion, NewsSignal) fire too often and get chopped. The ML model in `ml/models.py` currently predicts raw direction — throw that framing out. Instead, use each primary's own signals as the *triggering event* and train a meta-classifier on triple-barrier labels (López de Prado ch. 3.4).
>
> Specification:
> 1. In `ml/labels.py`, implement `triple_barrier_labels(prices, events, pt_sl=[1,1], vertical_barrier_days, volatility_target='ewma_20')` that returns {-1, 0, 1} for hit-stop, timeout, hit-target.
> 2. For each primary strategy, log every raw signal (including rejects) to `signals.parquet` with columns: ts, symbol, side, entry, atr, regime, session, strategy_name, confluence_score, news_sentiment, implied_vol_rank.
> 3. In `ml/meta_labels.py`, train a gradient-boosted meta-classifier (LightGBM) per strategy with target = "did this signal hit 1R before stop?" Use **purged K-fold with embargo = 2 × vertical_barrier** (ml/purged_cv.py already exists; reuse).
> 4. Calibrate probabilities with isotonic regression on a held-out fold.
> 5. Gate live entries: only take a primary signal when `meta_p_win > threshold` where threshold is chosen on OOS to maximize **cost-adjusted Sharpe**, not accuracy.
>
> Acceptance: OOS meta-filter lifts strategy PF to ≥ 1.3 with ≥ 40% signal rejection rate; Brier score < 0.22; calibration plot within 5% of diagonal across deciles.
>
> Deliver: backtest report comparing primary-only vs meta-filtered across 2 regimes (trending 2023, choppy 2024-H2).

---

### P2. Cost-aware training objective (stop optimizing accuracy)

> Objective: Replace classification accuracy as the model objective with a **PnL-aware loss** that penalizes false positives in proportion to realized slippage + commission + opportunity cost.
>
> Context: `train_ml_model_v2.py` optimizes log-loss. A model can be 60% accurate and still lose money after a 2-bp spread + 5-bp slippage. We need loss functions that match the economics.
>
> Specification:
> 1. In `ml/models.py`, add `PnLWeightedLoss(fee_bps=5, slip_bps=10, impact_coef=0.1)` — a custom objective for LightGBM that weights each sample by `|realized_r_multiple| - cost_in_r`.
> 2. Replace `classification_report` grading with: **expectancy per trade, profit factor, Sortino, Calmar, max adverse excursion, cost-drag %**.
> 3. Retain model selection across folds by **median OOS expectancy**, not mean — avoid single-fold luck.
> 4. Report cost-drag as a % of gross edge. If > 50%, reject the model.
>
> Acceptance: Cost-drag < 35%; median OOS expectancy > 0.15R/trade after costs across 5 folds; variance of Sharpe across folds < 0.4.

---

### P3. Regime-conditional strategy routing

> Objective: Wire the existing `regime_detector` (ADX, chop index, vol percentile) into a **strategy selector** that enables/disables strategies per regime and adjusts thresholds.
>
> Context: `pro_trading_config.yaml` already has `regime_strategy_map` — it's declarative but not enforced in the decision loop. `core/engine.py` currently runs all enabled strategies regardless.
>
> Specification:
> 1. At the top of each bar in `core/engine.py::_run_decision_cycle`, classify regime into one of: `trend_up_low_vol`, `trend_up_high_vol`, `trend_dn_low_vol`, `trend_dn_high_vol`, `range_tight`, `range_wide`, `chop`. 7 states, not 4 — low-vol trends require different thresholds than high-vol trends.
> 2. For each state, load a regime-specific config block (`regime_configs/*.yaml`) with: enabled_strategies, min_confluence, min_meta_p, position_size_multiplier, stop_atr_mult, tp_atr_mult.
> 3. Add a `RegimeStabilityGate` — require the regime to be stable for N bars before switching configs (avoid whipsaw on the regime signal itself).
> 4. Log every regime transition with reason vector (which indicator triggered) to `regime_transitions.jsonl`.
> 5. Forbid trading entirely in `chop` (already planned — enforce it).
>
> Acceptance: On the 2024 backtest, regime-routed version shows higher PF *and* lower trade count than ungated version. Transition logs readable for manual review.

---

### P4. Triple-barrier feature engineering (replace current labels)

> Objective: Kill the current `ml/labels.py` fixed-horizon labeling and replace with **event-driven triple-barrier** that respects volatility.
>
> Context: Current labels are "will price be up in N bars" — this trains the model to predict market direction, which is ≈ 50/50 and useless post-cost. We want to predict **which of the strategy's own entries will be profitable**.
>
> Specification:
> 1. `triple_barrier_labels(events, close, pt_sl, vertical, min_ret)` — events are the primary signal timestamps; barriers in multiples of trailing ATR(20); vertical barrier = 2× expected trade duration; drop samples with `|ret| < min_ret * atr`.
> 2. Apply **sample uniqueness weighting** (López de Prado 4.5): samples whose horizons overlap get down-weighted to avoid IID violations.
> 3. **Sequential bootstrap** when training — standard bootstrap is wrong for overlapping labels.
> 4. Add a `bar_type` option: `time`, `dollar`, `volume`, `imbalance`. Default to `dollar bars` (López de Prado 2.3) — time bars are statistically awful for ML.
>
> Acceptance: Training distribution is closer to IID (computed via AR(1) autocorrelation of residuals < 0.1); OOS metrics stable across bootstrap iterations.

---

# TIER 2 — Alpha Discovery

### P5. Cross-sectional momentum + mean-reversion stacking

> Objective: Add a **cross-sectional** signal layer (rank-based, market-neutral) on top of the current time-series signals. Single-stock bots leave enormous alpha on the table by ignoring relative strength.
>
> Specification:
> 1. Every 5 min, compute for all watchlist symbols: `rs_20 = (close / close.shift(20)) - 1`, `rs_ranked = rank(rs_20) / N`.
> 2. Generate `cross_sectional_mom` signal = top-quintile long, bottom-quintile short (if shorts enabled).
> 3. Overlay with short-horizon (5-bar) mean-reversion on residuals from cross-sectional regression: `r_i = alpha + beta * r_SPY + eps`; fade extreme `eps`.
> 4. Only trade a name if both **time-series** (primary strategy) and **cross-sectional** signals agree. Disagreement = no trade.
>
> Acceptance: Adding cross-sectional filter increases PF without increasing drawdown — confirmed on 2023–2024 backtest.

---

### P6. Options flow / put-call as smart-money regime signal

> Objective: `analysis/alternative_data.py` is a stub. Wire **equity put-call skew and unusual options volume** as a conviction adjuster.
>
> Specification:
> 1. Pull CBOE equity put/call ratio end-of-day; compute 20-day z-score. `pc_z > 1.5` → fear regime; `pc_z < -1.5` → complacency.
> 2. For each watchlist name, pull free options snapshots from Tradier/Polygon/yfinance. Compute: 30-day IV rank, 25-delta risk reversal, call/put volume ratio.
> 3. **Signal rules:** Call buying + IV compression + price consolidation = accumulation (+0.15 conviction on longs). Heavy put volume + IV spike + price pump = distribution (−0.20 conviction).
> 4. Expose as features to the meta-classifier, don't hard-code thresholds.
>
> Acceptance: Meta-model feature importance for options features in top 10; leave-one-out removal degrades OOS Sharpe by measurable amount.

---

### P7. News/sentiment with a real LLM, not TextBlob

> Objective: Replace the template-based sentiment widget with a **Claude-powered event classifier** that extracts: event type, surprise direction, magnitude, time horizon, and affected tickers — and debiases against headline sensationalism.
>
> Specification:
> 1. New module `analysis/llm_event_extractor.py` that batches recent headlines (60-second window) and calls Claude with this system prompt:
>
> ```
> You are a sell-side event analyst. For each headline, output strict JSON:
> {
>   "ticker": "...",
>   "event_type": "earnings|guidance|mna|regulatory|macro|analyst|product|legal|insider|other",
>   "direction": "bullish|bearish|neutral",
>   "magnitude_bps": <expected 1-day move in bps, -500 to 500>,
>   "horizon": "intraday|1-3d|1-4w|longer",
>   "novelty": <0-1, how much is this NEW information vs. already priced>,
>   "confidence": <0-1>,
>   "sensationalism_flag": <true if clickbait/speculative>,
>   "rationale": "<one sentence>"
> }
> Drop anything with novelty < 0.2 or sensationalism_flag = true.
> Do not hedge. If information is insufficient, output confidence < 0.3 and direction "neutral".
> ```
>
> 2. Cache results by headline hash; use Claude Haiku for cost.
> 3. Expose `magnitude_bps * confidence * novelty` as a signed feature to the meta-model.
> 4. Sentiment does **not** trigger trades alone — it conditions confluence on existing signals.
>
> Acceptance: Blind test on 100 historical headlines shows > 70% agreement with human labels; model latency < 2s per batch.

---

### P8. Microstructure features (intraday)

> Objective: Current features are mostly lagging OHLCV. Add microstructure features that predict short-term direction.
>
> Specification (add to `ml/features.py`):
> 1. **Amihud illiquidity** (|return| / dollar_volume) — 5/20/60 min windows.
> 2. **Roll spread estimator** from serial covariance of returns.
> 3. **VPIN** (Volume-Synchronized Probability of Informed Trading) on volume bars — predicts toxic flow.
> 4. **Order flow imbalance** proxy from signed tick-volume.
> 5. **Trade size distribution** — share of volume in prints > $500k (institutional footprint).
> 6. **Intraday seasonality residual** — subtract average-day profile from current-day profile.
>
> Acceptance: Adding these lifts meta-model AUC by ≥ 0.02 OOS; none of them are top-20 features alone (if they are, investigate leakage).

---

# TIER 3 — Risk & Execution Excellence

### P9. Volatility-targeted position sizing

> Objective: Replace fixed 1–3% risk per trade with **constant-expected-volatility** sizing. Current sizing traded TSLA and AAPL the same way — that's wrong.
>
> Specification (edit `risk/manager.py`):
> 1. Target daily portfolio vol: `portfolio_vol_target = 0.75%` (about 12% annualized, configurable).
> 2. Per-position vol contribution: `w_i = (target_vol / N_positions) / realized_vol_i * correlation_adjustment`.
> 3. Realized vol = EWMA-20 of squared returns, floor at 10th percentile of lookback to avoid infinite sizing when vol collapses.
> 4. Cap any single position at 5× its vol-parity allocation (prevents runaway sizing on stale-vol names).
> 5. Multiply by `regime_size_multiplier` and `meta_p_win_multiplier` from P1/P3.
>
> Acceptance: Rolling 60-day realized portfolio vol stays within ±30% of target. Drawdown on 2022 backtest reduced by ≥ 25% vs fixed-frac baseline.

---

### P10. Correlation-gated exposure

> Objective: `correlation_groups` in config is declarative only. Enforce it plus dynamic correlation.
>
> Specification:
> 1. At entry time, compute pairwise 60-day correlation between candidate and open positions on daily returns.
> 2. **Hard block** if adding this position pushes any pair correlation > 0.7 AND combined sector weight > 25%.
> 3. **Soft size-down** (0.5×) if cluster correlation > 0.5.
> 4. Track **effective number of bets** (Meucci): `N_eff = 1 / sum(w_i^2 * corr_adj)` — if N_eff < 2.5 with 5 positions, the portfolio is concentrated; reject new correlated adds.
>
> Acceptance: Max sector weight at any point in backtest ≤ 25%; N_eff ≥ 3 when portfolio is full.

---

### P11. Fractional Kelly with drawdown throttle

> Objective: The current Kelly floor of 50% is fine on paper but catastrophic in a drawdown. Implement drawdown-responsive Kelly.
>
> Specification:
> 1. `kelly_fraction = 0.25` base (conservative — full Kelly is suicide).
> 2. Multiply by `max(0.2, 1 - drawdown / max_acceptable_dd)` — at 50% of max DD, size is half; at max DD, size floor is 20%.
> 3. Recover linearly as equity recovers.
> 4. Log all size adjustments with reason vector.
>
> Acceptance: 2022 drawdown scenario shows smoother recovery; time-to-recovery < baseline.

---

### P12. Execution algo: participate-of-volume bracket

> Objective: Current `_place_bracket_orders` uses single limit orders. For larger sizes, this leaks edge. Implement a simple **POV** (participate of volume) child-order scheduler.
>
> Specification:
> 1. `execution/pov_executor.py` — given a parent order of size Q, target participation rate p (default 8% of 1-min volume), slice into child limits at mid±spread/4, refreshed every 15 s.
> 2. Fallback to market cross if unfilled by the signal's invalidation time (stop or TP must still fire).
> 3. Log every child order with slippage vs arrival-price benchmark. Track **implementation shortfall** per parent.
> 4. Only enable for orders > $25k notional; smaller orders stay single-limit.
>
> Acceptance: Average implementation shortfall < 3 bps on slippage-sensitive tickers.

---

# TIER 4 — Live Learning & Monitoring

### P13. Champion / challenger with shadow trading

> Objective: Never hot-swap a model. Run **challenger alongside champion** on live data for ≥ 30 trading days before promoting.
>
> Specification:
> 1. Store every champion prediction AND challenger prediction per bar (`shadow_predictions.parquet`).
> 2. Compute paper-PnL for the challenger's hypothetical trades with realistic costs.
> 3. Promotion criteria: challenger OOS Sharpe > champion Sharpe × 1.1 AND max drawdown ≤ champion × 1.2 AND minimum 100 simulated trades.
> 4. Dashboard widget shows both equity curves; promotion requires manual confirmation.
>
> Acceptance: No silent model replacements; shadow results audit-ready.

---

### P14. Concept drift detection (ADWIN + PSI)

> Objective: The current `ConceptDriftDetector` is rudimentary. Implement proper drift tests.
>
> Specification (in `ml/drift.py`):
> 1. **ADWIN** (Adaptive Windowing) on the rolling log-loss residual series — detects change-point in performance.
> 2. **Population Stability Index (PSI)** per top-20 feature vs training reference — `PSI > 0.2` = significant drift.
> 3. **Kolmogorov–Smirnov** on predicted probability distribution vs training.
> 4. When any trip: degrade to 0.5× size, alert, force retrain within 5 sessions.
> 5. Publish drift metrics to the dashboard; weekly drift report in email.
>
> Acceptance: Synthetic regime-shift injection triggers ADWIN within 200 samples with < 5% false-alarm rate.

---

### P15. Post-trade attribution & decision journal

> Objective: Every closed trade gets a structured post-mortem — not for nostalgia but for **identifying systematic failure modes** the meta-model can learn from next cycle.
>
> Specification:
> 1. On trade close, generate `trade_postmortem.jsonl` entry with: entry context snapshot, exit reason, realized R, max favorable excursion, max adverse excursion, time-in-trade, regime at entry vs exit, slippage breakdown, meta-prob at entry vs realized outcome, ranked contributing features (SHAP).
> 2. Weekly, cluster losses by feature profile (k-means on SHAP vectors) → find the "kind of losses" the model keeps taking.
> 3. Feed clusters back to the retraining loop as up-weighted samples.
>
> Acceptance: Weekly report identifies ≥ 1 actionable failure cluster; subsequent retrain reduces loss rate in that cluster.

---

# TIER 5 — LLM Commentary Brain (replace templating with Claude)

### P16. Master commentary system prompt

> Objective: Replace the template-based `CommentarySystem` with a Claude-driven narrative engine that produces **calibrated, non-hedging, audit-ready** trading commentary.
>
> Implementation: `core/llm_commentary.py`. Use Claude Sonnet 4.6 for depth on trade entries/exits; Haiku 4.5 for frequent heartbeats. Enable prompt caching on the static system prompt block.
>
> **System prompt — trade-entry narrator:**
> ```
> You are the quantitative PM for a systematic equities strategy, narrating a live trade decision for the decision journal. You have 15 years of systematic trading experience and a physics PhD. You speak tightly, like an institutional note, not a Reddit post.
>
> Your audience is (a) the trader reviewing why the bot entered, (b) a compliance reviewer auditing the decision, (c) your future self after the trade closes. Write for all three simultaneously.
>
> RULES OF ENGAGEMENT:
> 1. Never hedge with weasel words ("might", "could", "possibly"). Replace with numeric probabilities or magnitudes.
> 2. Every claim must be tied to a specific input in the evidence block. If you cannot point to a feature, do not make the claim.
> 3. State the disconfirming evidence — what would make this trade wrong, what the bear case is, what the nearest invalidation level is.
> 4. No platitudes ("the market is volatile today"). Only information that changes the decision.
> 5. No post-hoc rationalization. If conviction is low, say so and explain why we took it anyway (or why we shouldn't have).
> 6. End every entry note with a falsifiable prediction: "If price closes below $X within N bars, this thesis is wrong."
>
> OUTPUT STRUCTURE (strict):
> - THESIS (1 sentence): side, instrument, horizon, expected R-multiple.
> - EVIDENCE (3-5 bullets): each bullet cites a feature/value/regime fact. No narrative filler.
> - CONFLUENCE SCORE: numeric, 0-10, with the formula shown.
> - RISKS (2-3 bullets): specific conditions that invalidate.
> - INVALIDATION: exact price level and time window.
> - POSITION RATIONALE: why this size, not larger or smaller.
> - META-PROBABILITY: the model's win probability AND your sanity check ("this feels right" / "this feels optimistic because X").
>
> FORBIDDEN:
> - Emojis, exclamation marks, market-pundit clichés.
> - Reciting features the reader already sees in the data block.
> - Recommending actions — you narrate the bot's action, you don't override it.
>
> You will receive a structured JSON evidence block. Produce the commentary. Nothing else.
> ```
>
> **System prompt — trade-exit analyst (different model call):**
> ```
> You are conducting the post-mortem of a closed trade for the decision journal. Your job is to extract the LESSON, not to recap events.
>
> Required output:
> 1. OUTCOME: realized R, vs expected R at entry, vs model's meta-probability.
> 2. WHAT THE MODEL GOT RIGHT: specific features that aligned with outcome.
> 3. WHAT THE MODEL GOT WRONG: specific features whose SHAP contribution mispredicted.
> 4. REGIME ATTRIBUTION: did regime shift during the trade? how did it affect outcome?
> 5. EXECUTION ATTRIBUTION: realized slippage vs modeled; was the exit timely?
> 6. LESSON: one sentence. Something that could change a future decision. If there is no generalizable lesson, say "idiosyncratic outcome, no update."
> 7. RETRAIN FLAG: should this trade be up-weighted in the next retrain? (true/false + reason)
>
> You are blunt. You are not comforting anyone. If the bot was lucky, say so.
> ```
>
> **System prompt — pre-market daily plan (once per session):**
> ```
> You are the strategy's morning strategist. Before the open, given last night's close, overnight macro, economic calendar, and the current regime state, publish THE DAY'S PLAN.
>
> Output:
> - REGIME READ (trending / ranging / choppy / event-risk): confidence, what would change it.
> - WATCHLIST TIERING: A-list (high meta-prob), B-list (watch), blacklist (do-not-touch today, with reason).
> - KEY LEVELS per A-list name: nearest support, resistance, volume node, round number.
> - CATALYSTS: earnings, Fed, CPI, FOMC — with expected vol expansion.
> - PLAYBOOK: if regime X happens, we do Y. If Z happens, stand aside.
> - RISK BUDGET FOR THE DAY: max R at risk, max trade count, stop-trading triggers.
>
> Length: tight. This is a pre-market note, not a blog post.
> ```
>
> Acceptance:
> - Every trade has an entry note, exit postmortem, and daily plan.
> - All stored in `decision_journal.jsonl` with structured metadata + narrative.
> - Token cost < $10/day at current trade frequency (verify via caching).

---

### P17. LLM-as-conviction-adjuster (risky — gate carefully)

> Objective: Use Claude as a **tie-breaker**, not a decision-maker, when meta-probability is in the 0.45–0.60 grey zone.
>
> Specification:
> 1. When `0.45 ≤ meta_p ≤ 0.60`, call Claude with full evidence block and ask for a conviction adjustment in `{-0.10, -0.05, 0, +0.05, +0.10}` with written rationale.
> 2. **Guardrails:** LLM can only adjust within ±0.10; can never flip direction; can never override regime gates or risk limits.
> 3. Track LLM-adjustments separately in the journal. After 100 trades, measure if LLM adjustments improved expectancy. If not, disable this feature.
> 4. Feature-flag it: `features.llm_conviction_adjuster: false` by default.
>
> Acceptance: A/B journal confirms or refutes LLM edge after 100 trades; no silent influence on trades without logging.

---

# TIER 6 — Interface & UX

### P18. Single-pane trader dashboard (FastAPI + HTMX or React)

> Objective: Replace the scattered commentary stream with a **trader-first dashboard** inspired by Bloomberg Launchpad + a decision journal.
>
> Specification (edit `templates/`, `core/websocket_manager.py`):
> 1. **Top strip:** account equity, daily P&L in $ and R, open risk (current heat %), regime badge with confidence, circuit-breaker status, kill-switch button.
> 2. **Left column:** live positions with per-position P&L, MFE/MAE, current meta-prob, stop/TP, time-in-trade, one-click flatten.
> 3. **Center:** candlestick chart (lightweight-charts.js) with entry/exit markers, stop/TP lines, key levels from market_context module, regime bands colored.
> 4. **Right column:** live decision journal — every LLM note, collapsible, searchable, tagged by symbol/strategy/regime.
> 5. **Bottom strip:** equity curve (live vs. benchmark), rolling Sharpe, drawdown gauge, trade distribution histogram.
> 6. **Dark theme**, monospace numerics, green/red color-blind safe palette (use viridis, not raw red/green).
> 7. **Latency budget:** < 150 ms from bar close to dashboard update.
>
> Acceptance: The dashboard answers in ≤ 2 seconds: "What is my current risk? Why did I enter NVDA? Is the model calibrated this week?"

---

### P19. Decision journal search & filters

> Objective: The journal is useless if you can't search it.
>
> Specification:
> 1. Index `decision_journal.jsonl` into SQLite with FTS5 on narrative fields + structured columns for symbol, strategy, regime, outcome_r, meta_p, date.
> 2. Dashboard tab: filter by symbol × regime × outcome; group by failure cluster (from P15); click into the original entry/exit notes.
> 3. "Show me every losing trade in choppy regime where meta_p > 0.6" should be one click.
>
> Acceptance: Sub-100ms query on 10k journal entries.

---

### P20. Operator "explain this trade" command

> Objective: CLI/dashboard command: `explain TRADE_ID` returns a plain-English walk-through *generated on demand*, pulling evidence, meta-model SHAP, LLM notes, and outcome.
>
> Specification:
> 1. Endpoint `/api/explain/{trade_id}` — aggregates entry evidence, live notes, SHAP for entry prediction, execution log, exit postmortem. Passes to Claude with a "reconstruct the story" prompt.
> 2. Output is printable — intended for end-of-month compliance review.
>
> Acceptance: One operator can review 50 trades/hour using this tool.

---

# Orchestration Plan (suggested sequence)

1. **Week 1–2:** P2, P4 (cost-aware objective + triple-barrier labels) — breaks current training assumptions cleanly.
2. **Week 3–4:** P1 (meta-labeling) on top of re-trained features. Biggest expected lift.
3. **Week 5:** P3 (regime routing) — requires P1 to have meaningful thresholds.
4. **Week 6:** P9 + P10 + P11 (risk and sizing) — necessary before scaling size.
5. **Week 7–8:** P5, P6, P8 (alpha discovery). P7 (LLM sentiment) can run in parallel.
6. **Week 9:** P12 (execution algo) — only matters once edge is positive.
7. **Week 10:** P13, P14, P15 (monitoring/drift/postmortem).
8. **Week 11–12:** P16–P20 (LLM commentary + dashboard).

---

# Validation gates (apply to every change)

Before any prompt's changes are merged:
- **Purged walk-forward backtest** with realistic costs (spread from `data_providers`, commission from config, 0.1% market impact).
- **Monte Carlo bootstrap** of the equity curve — 1000 reshufflings, report 5th-percentile drawdown.
- **Regime-stratified** metrics (PF in trending vs choppy separately) — don't average across regimes.
- **Shadow trade** for ≥ 20 sessions before live promotion.
- **A/B journal** — every enabled feature is tagged, and we can measure its marginal contribution.

---

## How to use this pack

Copy any single `P#` block into Claude Code (with the role framing at top) and say: *"Implement this. Start with a plan, get my sign-off, then write the code with tests and a backtest report."* Each prompt is self-contained and references the real files and modules in this repo.
