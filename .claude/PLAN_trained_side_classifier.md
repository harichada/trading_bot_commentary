# Plan — Trained Side Classifier (RTX 3090)

**Status**: DRAFT for operator review (2026-05-13).
**Companion to**: `.claude/PLAN_symbol_side_classifier.md` (rule-based v1).
**Hardware target**: single RTX 3090 (24 GB VRAM, 10,496 CUDA cores,
GA102, sm_86).

This doc proposes a **trained** side-selection layer on top of the
rule-based classifier from the companion doc. Both ship. The rule-based
layer enforces the *hard gates* (earnings blackouts, HTB, halts, parabolic
regimes — things you don't want a model to override). The trained model
replaces the *soft scoring* — the linear-combination heuristic in §5.3 of
the companion doc — with learned weights tuned on labeled history.

The two layers compose:

```
final_allowed_sides = rule_based_hard_gates(symbol) ∩ trained_soft_scoring(symbol)
```

If the rule layer says `∅` for any reason, that wins. The trained layer
can never *expand* what the rule layer allowed — only *narrow* it.

---

## 1. Why both layers (not just trained)

A model is great at picking up subtle multivariate patterns. A model is
*terrible* at "this stock has earnings in 12 hours, do not touch it."
The rule layer handles structural/event constraints that don't belong in
training data (binary, sparse, asymmetric-consequence). The trained
layer handles the gradient of evidence between trend, sentiment, volume,
and cross-asset signals — patterns the linear model in the companion
doc gestures at but doesn't really learn.

**Operator rationale**: the rule layer is *auditable* — every decision
has a one-line reason traceable to a yaml-configurable rule. The trained
layer is *adaptive* — it picks up regime shifts that the rule layer
can't. Both properties matter; neither alone is enough.

## 2. Model architecture — three real options + recommendation

Each option is evaluated on (a) fit for the problem, (b) RTX 3090
utilization, (c) inference latency for 30 symbols / 30 sec, (d) operator
ability to debug it.

### Option A — 1D-CNN over engineered feature time series

- **Input**: per-symbol tensor `(timesteps=60, features=F)` — 60 daily
  bars × F engineered features (the rule-based feature set + raw OHLCV
  normalized, indicators, sector ETF rel return, SPY rel return, VIX).
- **Architecture**: 3-4 dilated conv blocks → global pool → 2-layer MLP →
  softmax over `{LONG_wins, SHORT_wins, NEITHER}` or two sigmoid heads.
- **Parameters**: ~2 M.
- **Training time on 3090**: ~10 min per epoch on 5 years of US-equity
  watchlist data; total run ~2–4 hours.
- **Inference**: <1 ms per symbol in batch; trivial on 3090.
- **Debuggability**: medium — gradient saliency maps work, attention
  doesn't exist.

### Option B — Temporal Fusion Transformer (TFT)

- **Input**: same engineered series + categorical embeddings (sector,
  symbol-id, day-of-week, regime).
- **Architecture**: variable-selection networks per feature → LSTM
  encoder → multi-head attention → quantile output heads.
- **Parameters**: ~5–10 M.
- **Training time on 3090**: ~30 min per epoch; total ~6–12 hours.
- **Inference**: ~5 ms per symbol in batch; well within budget.
- **Debuggability**: high — TFT exposes per-feature variable importance
  AND per-timestep attention weights. Critical for an operator who has
  to trust the output.
- **Reference impl**: `pytorch-forecasting` library (mature, on Lightning).

### Option C — PatchTST (patch-based time series transformer)

- **Input**: each symbol's 1D series split into overlapping patches;
  patches become "tokens" for a standard transformer encoder.
- **Architecture**: patch embed → 6-layer transformer → linear head.
- **Parameters**: ~3–5 M.
- **Training time on 3090**: ~20 min per epoch; total ~4–8 hours.
- **Inference**: ~3 ms per symbol; fine.
- **Debuggability**: lower than TFT (patch attention is harder to
  interpret).
- **Reference impl**: `nixtla/neuralforecast`, also custom (~150 LoC).

### Recommendation

**Option B (TFT)** for v1. Reasons:

1. **Interpretability is non-negotiable for a trading agent.** TFT's
   variable-selection network produces per-feature importance scores
   that we can log alongside each decision. When the model says "SHORT
   only" for a symbol and you want to know why, TFT can answer ("70%
   of the decision weight came from sma_alignment_down and
   rs_vs_sector_5d_negative"). CNN/PatchTST cannot.
2. **Native handling of categorical features.** Sector, symbol-id,
   regime-bucket all become learned embeddings — better than one-hot.
3. **Quantile outputs**, not point estimates. Output is `P10`, `P50`,
   `P90` of the forward return — calibrated uncertainty per side.
   Lets us require both *high P50* AND *narrow spread* for a
   confidence-gated decision.
4. **Battle-tested implementation** via `pytorch-forecasting`. We are
   not writing the architecture from scratch; we're writing the data
   pipeline + the operator-facing wrapper.

PatchTST is a strong fallback if TFT training stability becomes an
issue (TFT is fussier).

## 3. Label design — the hardest part

This is where most ML-for-trading projects go wrong. The model's
behavior is determined by what we ask it to predict.

### 3.1 Candidate label schemes

| Scheme | Definition | Pros | Cons |
|---|---|---|---|
| Forward return at H | sign of return H bars ahead | Simple | Ignores path: a +5% with a -10% drawdown is "good" |
| Triple-barrier (Lopez de Prado) | label = first barrier hit (TP, SL, time) | Captures path-dependent reality | Requires choosing per-symbol barriers |
| Meta-labeling | base predictor + a "should I take this trade" model | Already used elsewhere in this repo (`tests/test_meta_labels.py`) | Adds a stage |
| Sharpe of forward window | risk-adjusted return over [t, t+H] | Continuous, smooth | Sensitive to window choice |

### 3.2 Recommendation: triple-barrier with adaptive ATR-based barriers

For each (symbol, day) example, look forward H=5 trading days. Set:

- Upper barrier = `entry + 1.5 × daily_ATR_14`
- Lower barrier = `entry - 1.5 × daily_ATR_14`
- Time barrier = 5 trading days

Label is:
- `LONG_wins` if upper hit first
- `SHORT_wins` if lower hit first
- `NEITHER` if time barrier hits first (range / chop)

This labeling is what the strategy actually cares about — does a side
*work* under realistic stop/target geometry? — and it's already
implemented in this repo's `tests/test_labels.py` so the labeling
function exists.

### 3.3 Why not "predict return"

Predicting a continuous return is harder, has no clean threshold for
"trade vs don't", and historically performs worse than predicting the
classification you actually act on. The triple-barrier label IS the
decision the bot makes; train on the decision.

## 4. Feature set

Two channels: **dynamic** (changes daily) and **static** (per-symbol).

### 4.1 Dynamic features (per timestep — daily bars, 60-day window)

All of the rule-based features from `PLAN_symbol_side_classifier.md`,
plus:

- Raw OHLCV (log-returns and normalized)
- TA: RSI_14, MACD, ADX, BB_width, ATR_14 — same set the existing
  strategies use
- SPY relative: `(symbol_return − SPY_return)` per day
- Sector ETF relative
- VIX level + 5-day change
- News-day flag (1 if any article today, 0 otherwise)
- Aggregated daily sentiment score
- Volume z-score vs 20-day avg

### 4.2 Static features (per symbol, embedded)

- Sector (one of 11 GICS sectors)
- Symbol ID (learned embedding, ~32-dim — captures "this is NVDA-like")
- Float-size bucket (micro / small / mid / large / mega)
- Average daily $ volume bucket
- Optionable / shortable flags (with HTB as separate hard gate)

### 4.3 Known-future features (TFT supports these)

For the forward window the model is predicting over:
- Day of week
- Is earnings day inside window? (binary)
- Days to next CPI / FOMC date
- Triple-witching day flag

TFT's variable-selection network handles known-future features cleanly
— they're available at prediction time, just not at training-feature
time.

## 5. Training data

### 5.1 Source

- **OHLCV**: yfinance (free, deep history) for backfill 2015 → today;
  Schwab REST for daily refresh after that. yfinance is fine for
  *training* data — minor data-quality issues are washed out by volume.
- **News**: existing `news_aggregator` cache + Alpaca historical news
  endpoint for backfill where the bot's cache is too shallow.
- **Sector ETFs**: 11 SPDR sector ETFs (XLE, XLK, XLF, XLY, XLV, XLI,
  XLP, XLU, XLB, XLRE, XLC) from yfinance.
- **VIX**: yfinance ticker `^VIX`.

### 5.2 Universe

S&P 500 + Russell 1000 + the bot's own historical watchlist names
(captured by `screener.top_movers`). De-duplicated, ~1,200 symbols.

### 5.3 Volume

- Universe: ~1,200 symbols
- History: 8 years × 252 trading days = ~2,000 examples per symbol
- Total: ~2.4 M training examples before filtering
- After purging (events, halts, illiquid days): ~1.8 M

That's plenty for a 5-10 M-parameter model.

### 5.4 Splits — purged time-series CV

**Critical**: the existing repo already has purged-CV scaffolding
(`tests/test_purged_cv.py`). Reuse it. Standard k-fold is wrong for
financial data — leakage destroys reported accuracy.

Splits:
- Train: 2017-01 → 2024-12
- Embargo: 5 trading days (no examples from this window)
- Validation: 2025-01 → 2025-06
- Embargo: 5 trading days
- Test (held out, never tuned against): 2025-07 → 2026-05

The 5-day embargo prevents label leakage from triple-barrier outcomes
that span the split boundary.

## 6. Hardware utilization — RTX 3090 specifics

| Workload | Resource | Headroom |
|---|---|---|
| TFT training (~10 M params, batch 256, 60 timesteps × ~80 features) | ~6 GB VRAM | 4× headroom for bigger batches or longer history |
| Per-epoch time | ~25-35 min on 1.8 M examples | Total run ~6-12 hours; can run overnight |
| Inference batch (30 symbols × 60 timesteps × 80 features) | ~50 MB VRAM, <10 ms | Inference is free on this card |
| Optional: large news-LM embedding (FinBERT-small) | ~2 GB VRAM resident | Fits alongside main model |
| Optional: ensemble of N models | ~6 GB × N | Can run 3-4 models simultaneously |

**Plenty of headroom.** The 3090 is over-spec for v1; that's a *feature*
because it means:
- Inference latency is never a bottleneck (always <100 ms even with
  large models)
- Can train multiple seeds / multiple horizon heads in parallel
- Room for a news-language-model side branch (FinBERT or similar) if
  v1 leaves edge on the table

### 6.1 Software stack

- CUDA 12.x (already required by recent PyTorch)
- PyTorch 2.4+ with CUDA support (`pip install torch --index-url ...`)
- `pytorch-lightning` for training loop / checkpointing / logging
- `pytorch-forecasting` for TFT implementation
- `pyarrow` + parquet for efficient training-data storage
- `mlflow` or `tensorboard` for experiment tracking — I'd default to
  TensorBoard (zero-config, just `tensorboard --logdir runs/`)

### 6.2 Operational footprint

- Model artifact: ~30 MB on disk (TFT, fp32)
- Quantized inference artifact: ~10 MB (fp16 or int8); not needed
  for v1 given the 3090 is idle most of the time
- Trained model versioned via the existing `ml_model_manager_safe.py`
  pattern (`model_v_2026-05-13_154212.pt`)

## 7. Serving — where the model lives in the bot

```
core/
├── symbol_classifier.py          ← rule-based (companion plan)
├── trained_classifier/
│   ├── __init__.py
│   ├── model.py                  ← TFT wrapper, load + predict
│   ├── features.py               ← feature builder (pure)
│   ├── data.py                   ← yfinance/Schwab daily-bar loader + cache
│   └── checkpoint_loader.py      ← lazy GPU-load on first call
├── side_classifier.py            ← THE COMPOSER. Calls both, combines, returns final SymbolSideDecision
```

`side_classifier.py` is the single entry point the engine calls. It:

1. Calls `symbol_classifier.classify(sym)` (rule-based; <1 ms, no GPU).
2. If rule returns `∅`, return `∅` immediately. **Model never overrides hard gates.**
3. Otherwise call `trained_classifier.predict(sym)` (GPU; ~5 ms).
4. Intersect: `final = rule.allowed_sides ∩ trained.allowed_sides`.
5. Log both decisions + the intersection at WARNING level if they
   disagree (so operator sees friction).

### 7.1 Model lifecycle

- **Load on first signal**: model weights stay on GPU between calls; lazy
  load on first prediction after engine start.
- **Reload-on-disk-change**: `inotify`-style watcher on the model
  artifact directory. New model dropped → atomic swap on next inference.
- **Fallback**: if GPU not available (3090 unplugged, CUDA driver mismatch)
  or if model load fails, `trained_classifier.predict()` returns
  `{LONG, SHORT}` (allow all) and logs WARNING. Bot stays operational on
  rule-based layer only. **Fail-open here** because the rule layer is
  itself fail-closed; defense in depth.

### 7.2 Inference path

- Maintained cache: `{symbol → (decision, expires_at)}` with 30-min TTL
  (same as rule-based layer, for the same reason).
- Batch inference: when N symbols expire simultaneously, batch them
  into one forward pass.
- Pre-warm: on engine startup, queue all current watchlist + position
  symbols for inference before the first signal evaluation.

## 8. Validation — same shadow → advisory → live discipline

Identical posture to the rule-based plan + the news verifier history:

| Phase | What's enabled | Duration |
|---|---|---|
| 0 — Data backfill | Build 8-year training set | 1-3 days (mostly yfinance throttling) |
| 1 — Train v1 | TFT on train split, evaluate on validation | ~1-2 days |
| 2 — Held-out test | Score on test split, never look at train/val again. Report: confusion matrix, per-side precision/recall, calibration plot, equity-curve simulation | 1 day |
| 3 — Shadow live | Model loaded, predicts on every signal, logs decisions; gate is OFF. Compare to operator judgment overnight | ≥ 5 sessions |
| 4 — Advisory live | Predictions logged AND surfaced on dashboard but still no gating. Operator clicks through commentary cards | ≥ 5 sessions |
| 5 — Live gating | Composer (`side_classifier.py`) intersects rule + model; intersection gates entries | indefinite |

Pause / rollback at any phase. Same pattern as `v-news-verifier-advisory-2026-05-08`.

### 8.1 Required metrics on the held-out test set

- **Per-side precision** at threshold 0.55: of all entries the model allowed, what fraction were profitable per the triple-barrier label?
- **Calibration**: when model says P(LONG_wins) = 0.7, does that side actually win 70% of the time? Reliability diagram.
- **Equity-curve simulation**: bot's existing entry/exit logic + model gate → simulated P&L on the test period. Net Sharpe, max drawdown, # trades.
- **Per-regime breakdown**: split test set by VIX regime (low / med / high); confirm model isn't only working in bull markets.
- **Comparison baseline**: rule-based-only equity curve on the same test period. Trained model must beat it net of inference cost.

If any of these metrics fail (e.g., precision below 55%, calibration
visibly off, equity curve worse than rule-based baseline), v1 doesn't
ship to shadow. Iterate.

## 9. Continuous training — drift management

Markets change. A 2017-2024 trained model will degrade. The plan:

- **Weekly retrain** on the trailing 4-year window, triggered Sunday
  00:00 by a `cron` task.
- **Drift monitor**: nightly job compares yesterday's predictions
  against realized triple-barrier outcomes. If precision on the past
  30 sessions drops below 50% (vs ~60% target), page operator.
- **Champion-challenger**: new model trains in shadow alongside current
  champion. Promotion to live requires matching or beating champion
  on the trailing 5 sessions of advisory predictions.

## 10. Open questions for operator review

These need answers before implementation starts:

1. **CUDA / PyTorch already installed on the bot host?** The 3090 needs
   the right driver + CUDA runtime. Confirm `nvidia-smi` runs on the
   bot machine; confirm `python -c "import torch; print(torch.cuda.is_available())"`
   returns `True`. If not, that's a one-time install step.
2. **Latency budget at signal time.** The bot's existing analysis loop
   runs every 30 s during RTH and evaluates ~30 watchlist symbols.
   Adding ~5 ms × 30 = 150 ms of GPU inference is negligible. Are you
   ok with that cost? (Almost certainly yes; flagging for completeness.)
3. **Bot machine ≠ training machine?** If the bot is on a different
   host than the 3090, we need a path to copy the trained artifact
   from training host to bot host (rsync, S3, etc.). If same machine,
   trivial.
4. **Newsletter-style training reports.** Each retrain produces a
   markdown report (metrics, calibration plots, feature importance
   table). Where should those live — `docs/training_runs/` or some
   external dashboard? Default proposal: `docs/training_runs/YYYY-MM-DD.md`.
5. **Feature-importance gating.** TFT exposes per-feature attention.
   Want a guardrail like "if a feature's importance jumps >2σ vs prior
   weekly retrain, flag for operator review" — to catch the model
   suddenly relying on a single noisy feature? My take: yes, with a
   WARNING log line, but not a hard block.
6. **Risk appetite for the rollout cadence.** Phases 3-5 are each ≥ 5
   sessions = ~2 weeks per phase = 6+ weeks before the model is
   live-gating. Faster acceptable? Slower?
7. **Continuous training compute window.** Weekly retrain is ~6-12
   hours. Sundays during US-equity-closed window is the obvious slot.
   Confirm the 3090 is free at that time.

## 11. What ships in what order

```
Phase 0: Companion rule-based classifier (`PLAN_symbol_side_classifier.md`)
         ships first. Standalone. Live-gating after its own shadow → advisory.

Phase 1: Trained-classifier scaffolding — data pipeline + features + TFT
         training code. Producing a trained model is the deliverable; not
         wired to anything live yet.

Phase 2: Composer (`core/side_classifier.py`) lands. Wired to BOTH classifiers.
         Default mode: rule-based gates only; trained side runs but its
         output is logged-only.

Phase 3: Trained side moves through shadow → advisory → live gating
         (the 5-phase rollout in §8 above).
```

The rule-based classifier provides immediate value with zero ML risk.
The trained classifier is additive. If training never converges or the
model never beats the rule baseline, the bot still has the rule layer
and is in better shape than today. **Trained model is a bonus, not a
load-bearing component.**

## 12. What this is NOT — same discipline as the companion doc

- Not an end-to-end RL agent. Predicting `LONG_wins / SHORT_wins / NEITHER`
  is a clean supervised problem. RL on trading is a several-year project
  with worse expected outcomes for v1.
- Not a sentiment-only model. News is a feature, not the spine.
- Not autoML. Hand-pick TFT, hand-engineer features, log everything,
  ship the simplest thing that beats the baseline.
- Not a replacement for any existing filter. Falling-knife, late-entry,
  price-direction guards in `strategies/news_strategy.py` all stay. The
  trained classifier is upstream of them and cannot rescue a signal
  they already rejected.
- Not an excuse to disable the rule-based layer. They compose. Forever.

---

## Summary of what you're approving (if you approve)

1. A trained TFT model that scores per-symbol `(long_score, short_score)`
   on 60-day daily feature windows, trained on triple-barrier labels.
2. A composer (`core/side_classifier.py`) that intersects rule-based and
   trained decisions; rule-layer hard gates always win.
3. RTX 3090 utilization: ~6 GB VRAM training, <10 ms inference, weekly
   retrain on Sunday.
4. A 5-phase rollout identical in posture to the rule-based plan and
   the news verifier history: shadow → advisory → live, with the option
   to halt at any phase if metrics misbehave.
5. Continuous-training infrastructure (cron weekly + drift monitor +
   champion-challenger).
6. A clear "this is additive, not load-bearing" framing: if training
   never converges, bot still runs on rule-based layer.

Tell me what to redirect.
