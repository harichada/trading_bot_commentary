# Composite View Architecture

> Architectural Decision Record (ADR) for the per-symbol holistic assessment service that inverts the bot's decision flow from strategy-first to context-first.

| Field | Value |
|---|---|
| **Status** | Proposed — pending operator approval |
| **Author** | Operator + AI advisory |
| **Date** | 2026-06-09 |
| **Target ship** | Week 1 (foundation), Week 2 (gate wiring), Week 3 (full integration) |
| **Supersedes** | (none — additive layer) |
| **Related work** | MarketContext (2026-06-08), Direction Reader (2026-05), Side Classifier (designed 2026-05) |

---

## Table of Contents

1. [Problem statement](#1-problem-statement)
2. [Goals and non-goals](#2-goals-and-non-goals)
3. [Constraints](#3-constraints)
4. [Architecture overview](#4-architecture-overview)
5. [Data model](#5-data-model)
6. [Component breakdown](#6-component-breakdown)
7. [Public API](#7-public-api)
8. [Integration points](#8-integration-points)
9. [Failure modes and resilience](#9-failure-modes-and-resilience)
10. [Performance budget](#10-performance-budget)
11. [Security](#11-security)
12. [Testing strategy](#12-testing-strategy)
13. [Observability](#13-observability)
14. [Migration plan](#14-migration-plan)
15. [Risks and mitigations](#15-risks-and-mitigations)
16. [Alternatives considered](#16-alternatives-considered)
17. [Open questions](#17-open-questions)
18. [Glossary](#18-glossary)

---

## 1. Problem statement

### 1.1 What's broken today

The bot is **strategy-first**: four strategies (mean-reversion, breakout, momentum, news) independently scan every symbol on every bar, each through a narrow lens (RSI > threshold, close > BB band, sentiment > threshold, etc.). Each strategy produces signals based on its own pattern-match without holistic awareness of the symbol's broader context.

Filters (correlation guard, falling-knife, rising-peak, price-direction, falling-knife-news, MarketContext gate) reject the obvious bad signals AFTER they fire. This works for *some* failure modes but is reactive, not proactive.

### 1.2 Real failures this approach produced

| Date | Symbol | Outcome | Root cause |
|---|---|---|---|
| 2026-06-09 09:42 | **SMCI** | Mean-rev LONG knife-caught into −$157.99 | Strategy fired on RSI 28.6 + close < BB lower. Same bar: `close $43.33 < SMA50 $44.36`, `MACD -0.23` — every other signal said *falling knife*. Strategy didn't ask. |
| 2026-06-09 ~10:30 | **PAVS** | Would have shorted at $5.17, stopped at $7.71 (+49% catastrophic move) | Earlier same session, PAVS's `bb_lower` went *negative* (mathematically broken stock). One strategy flagged `invalid_indicators`; the next signal on the same name proceeded as normal. No persistence of the "this stock is degraded" finding. |
| 2026-06-09 ~10:00 | **CRDO/DKNG/SJM** | 3 out of bot's would-have SHORT candidates stopped out | Strategies fire at *rally peak* (RSI > 70 + close > BB upper). No mechanism to wait for *post-peak rollover confirmation*. |
| Recurring | All overbought names | 21 SHORT candidates today, ~4 stops vs 3 wins (mixed) | Backtest's PF 1.04 verdict materializing live: the strategy identifies right symbols but at wrong moments. |
| Recurring | Operator's manual closes | Operator routinely overrides bot timing for profit-take | Bot lacks the regime-level synthesis that operator does instinctively. |

### 1.3 What the operator actually wants

Direct quote (2026-06-09):

> "I am creating this bot to get rid of manually assessing the trend, I want the bot to assess in realtime which a normal human would miss, the price movement, the trend, the company financials, positive/negative news, the direction and its strength"

The operator's vision is **symbol-first**: for each name on the watchlist, compute a holistic view across five dimensions (technical, trend, financials, news, direction + strength). Strategies should consume this view, not generate competing opinions on the same raw data.

### 1.4 Why this matters now

Three converging signals:
1. **Empirical**: Today's 1-LONG-loser-plus-21-blocked-SHORTs shows the current architecture is hitting its ceiling.
2. **Strategic**: Operator articulated the vision explicitly today. Has the right framing.
3. **Tactical**: Existing components (MarketContext, Direction Reader, Side Classifier design) are 60% of the input layer. Adding the synthesis layer is finite work.

---

## 2. Goals and non-goals

### 2.1 Goals

1. **Per-symbol holistic assessment** computed continuously (every analysis bar) that synthesizes 7+ input dimensions into a single directional view + tradability score.
2. **Inversion of decision flow**: strategies fire only when the view supports them. No more "strategy says LONG, view says bear, trade happens anyway."
3. **Quality-over-quantity throughput**: expect 40-60% fewer trades but materially higher per-trade expected value.
4. **Explainability**: every view computation logs its component scores and the factors driving the direction call. No black-box decisions.
5. **Backward compatibility**: existing strategies continue to work; the view is a new gate, not a replacement.
6. **Empirical validation path**: A/B testable via config flag, sim-soakable, before live enablement.

### 2.2 Non-goals

1. **Not replacing existing strategies.** Mean-rev, breakout, momentum, news strategies remain. They become *consumers* of the view, not generators of independent decisions.
2. **Not eliminating operator override.** Manual close/open via dashboard or Schwab still works. The view informs the bot; the operator is still in control.
3. **Not building an ML predictor.** The view is rules-based and transparent. ML meta-labeling (existing) operates downstream as a confidence layer, not as the view itself.
4. **Not rebuilding the data layer.** Schwab stream, news aggregator, indicator computation all stay as-is. The view consumes them.
5. **Not aiming for "predict the market."** The view assesses *current* state holistically. It doesn't forecast.

---

## 3. Constraints

| Constraint | Implication |
|---|---|
| Must run in main analysis loop's per-tick budget | Composite view computation ≤50ms per symbol, cached with 60s TTL |
| Must not break existing functionality | Feature flag `ENABLE_COMPOSITE_VIEW_GATE` (default False until validated) |
| Must integrate with existing audit log | Each view computation emits structured `engine_decision` event |
| Must work with current python/dependencies | Pure Python, no new system dependencies; new pip packages OK |
| Operator's account is the only target | No multi-tenancy concerns |
| Single-process bot (no horizontal scaling) | In-process cache is fine; no Redis needed |
| Maintenance burden must be sustainable | One developer (operator + AI advisory); each component ≤200 LOC |
| **Single Schwab token, single Yahoo session** | Components must share existing fetchers; no parallel rate-limit collisions |

---

## 4. Architecture overview

### 4.1 Decision flow comparison

**Current (strategy-first):**

```
                ┌──────────────┐
                │  Schwab tick │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Indicators  │
                └──────┬───────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
   ┌───▼───┐       ┌───▼───┐       ┌───▼────┐
   │MeanRev│       │Breakout│      │ News   │
   │ scans │       │ scans  │      │ scans  │
   └───┬───┘       └───┬───┘       └───┬────┘
       │               │               │
       └────────┬──────┴───────┬───────┘
                │              │
                ▼              ▼
       (each fires signals based on its own pattern)
                │
       ┌────────▼─────────┐
       │  Signal Router   │
       │  + Filters (10+) │
       └────────┬─────────┘
                │
                ▼
        (most rejected, some pass)
```

**Proposed (symbol-first):**

```
                ┌──────────────┐
                │  Schwab tick │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  Indicators  │
                └──────┬───────┘
                       │
       ┌───────────────▼───────────────┐
       │   Composite View Service      │
       │   (per-symbol synthesis)      │
       │                               │
       │  Inputs:                      │
       │  - Trend (Direction Reader,   │
       │    MarketContext)             │
       │  - Momentum (MACD, RSI, ADX)  │
       │  - Volatility quality (ATR/$, │
       │    bb_lower sanity)           │
       │  - Sector (MarketContext)     │
       │  - News (aggregator+sentiment)│
       │  - Fundamentals (Yahoo)       │
       │  - Liquidity ($-volume)       │
       │                               │
       │  Output:                      │
       │  CompositeView {              │
       │    direction, strength,       │
       │    tradability, factors[]     │
       │  }                            │
       └──────────┬────────────────────┘
                  │
       ┌──────────▼────────────┐
       │  View Gate (per       │
       │  symbol, per bar)     │
       │                       │
       │  tradability < 0.4    │
       │   → skip ALL strategies│
       │                       │
       │  direction = bull     │
       │   → SHORT strategies  │
       │     skipped           │
       │                       │
       │  strength < 0.5       │
       │   → strategies skipped│
       │     (low conviction)  │
       └──────────┬────────────┘
                  │
       (pass-through to strategies for actionable names only)
                  │
       ┌───────────────┼───────────────┐
       │               │               │
   ┌───▼───┐       ┌───▼───┐       ┌───▼────┐
   │MeanRev│       │Breakout│      │ News   │
   │       │       │        │      │        │
   └───┬───┘       └───┬───┘       └───┬────┘
       │               │               │
       └────────┬──────┴───────┬───────┘
                │              │
                ▼              ▼
       (only fire when view supports them)
                │
       ┌────────▼─────────┐
       │  Signal Router   │
       │  + remaining     │
       │  risk/correlation│
       │  filters         │
       └──────────────────┘
```

### 4.2 The single most important change

**The View Gate sits between Indicators and Strategies.** Strategies no longer see every symbol on every bar. They see only the *actionable subset* the view determined is worth a strategy's attention given current direction and strength.

This is **not** a new filter on outputs — it's a gate on **inputs**. Strategies don't waste cycles on PAVS-like degraded names or SMCI-like falling knives because those names never reach them.

---

## 5. Data model

### 5.1 CompositeView dataclass

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional

@dataclass(frozen=True)
class CompositeView:
    """Per-symbol holistic assessment computed by the Composite View Service.

    Immutable: produced once per symbol per analysis bar (with 60s cache TTL).
    Strategies consume this read-only; no mutation, no side effects.

    Direction is a discrete trichotomy by design — the View Gate uses it as a
    hard filter (no "kinda bull" trades). Strength represents conviction in
    that classification. Tradability is the global "is this symbol even
    worth our attention right now" score.
    """
    symbol: str
    timestamp: datetime

    # Top-line synthesis
    direction: Literal['bull', 'bear', 'neutral']
    strength: float          # 0.0 (no conviction) .. 1.0 (extreme conviction)
    tradability: float       # 0.0 (skip entirely) .. 1.0 (high-quality candidate)

    # Audit trail — every component's contribution
    trend_score: float       # -1.0 (strong down) .. +1.0 (strong up)
    momentum_score: float    # -1.0 .. +1.0
    volatility_quality: float  # 0.0 (degraded data) .. 1.0 (clean)
    sector_score: float      # -1.0 .. +1.0 (sector relative strength)
    news_score: float        # -1.0 .. +1.0 (recent sentiment-weighted)
    fundamentals_score: Optional[float]  # -1.0 .. +1.0; None if no data
    liquidity_score: float   # 0.0 .. 1.0

    # Human-readable factor lists for commentary / dashboard
    bull_factors: tuple[str, ...]
    bear_factors: tuple[str, ...]

    # Hard veto reasons — non-empty means tradability is 0
    veto_reasons: tuple[str, ...]

    # Provenance — which input bars were used (cache invalidation hints)
    indicators_bar_ts: Optional[datetime] = None
    news_max_age_s: Optional[float] = None
    fundamentals_max_age_s: Optional[float] = None

    def allows_long(self) -> bool:
        """Convenience: would a LONG strategy be permitted by this view?"""
        return (self.tradability >= 0.4
                and self.direction == 'bull'
                and self.strength >= 0.5)

    def allows_short(self) -> bool:
        """Convenience: would a SHORT strategy be permitted by this view?"""
        return (self.tradability >= 0.4
                and self.direction == 'bear'
                and self.strength >= 0.5)
```

### 5.2 Threshold rationale

| Threshold | Default | Why |
|---|---|---|
| `tradability >= 0.4` | 0.4 | Below this is "data is broken or symbol is unreliable" — PAVS-killer threshold. Tunable. |
| `strength >= 0.5` | 0.5 | Below this is "directional call but not confident" — avoids marginal trades. Maps to the median of conviction calls in observed data. |
| `direction in {bull, bear, neutral}` | (discrete) | Trichotomy forces a clean decision. Strategies can't trade "kinda bull." Either commit or skip. |

All thresholds are config-driven (`Config.COMPOSITE_VIEW_TRADABILITY_FLOOR`, `Config.COMPOSITE_VIEW_STRENGTH_FLOOR`).

---

## 6. Component breakdown

Each component is a **pure function** that ingests raw inputs and returns a normalized score. Components are independent; one's failure doesn't cascade. Synthesis happens in `_synthesize_view()` after all components run (failures returning `None` are handled with weighted-average fallback).

### 6.1 Trend component

**Input**: Direction Reader output + MarketContext regime
**Output**: `trend_score: float` in [-1.0, +1.0]
**Logic**:
- Direction Reader's 5-component score normalized to [-1.0, +1.0]
- Multiplied by MarketContext regime alignment: risk_on adds +0.2 to bull, risk_off subtracts
- Falling-knife pattern (close < SMA50 AND MACD < 0) caps at -0.5
- Rising-peak pattern (close > SMA50 AND MACD > 0) caps at +0.5

**Existing code reused**: `core/direction_reader.py`, `core/market_context.py`

### 6.2 Momentum component

**Input**: MACD line + signal + histogram, RSI, ADX
**Output**: `momentum_score: float` in [-1.0, +1.0]
**Logic**:
- MACD slope direction: +0.3 if rising > 2 bars, -0.3 if falling
- RSI position: maps 50 to 0.0, 70+ to +0.5, 30- to -0.5
- ADX modulation: scores multiplied by min(ADX/25, 1.0) — weak-trend dampener

**Existing code reused**: indicator computation in `analysis/technical.py`

### 6.3 Volatility quality component

**Input**: ATR, price, bb_lower, bb_upper, last 20 bars' true range
**Output**: `volatility_quality: float` in [0.0, 1.0]
**Logic**:
- Veto if `bb_lower <= 0` ever in last hour (PAVS-killer)
- Veto if `ATR/price > 0.05` (5%+ daily volatility = meme territory)
- Veto if NaN in any of MACD, RSI, ADX
- Otherwise scaled: ATR/price 0.005 → 1.0, ATR/price 0.05 → 0.0

**New work**: persistent "this symbol degraded today" flag in symbol-state table

### 6.4 Sector component

**Input**: symbol's mapped sector ETF, MarketContext sector_strength
**Output**: `sector_score: float` in [-1.0, +1.0]
**Logic**: Directly use `MarketContext.sector_strength` already computed

**Existing code reused**: `core/market_context.py`

### 6.5 News component

**Input**: news aggregator output (last 4h) — sentiment + count + recency
**Output**: `news_score: float` in [-1.0, +1.0]
**Logic**:
- Weighted average of last 4h sentiments by recency (exponential decay, half-life = 30 min)
- Scaled by min(article_count/5, 1.0)
- Verification gate (existing v-news-verifier-2026-04) multiplies score by 0.5 if verifier disagrees

**Existing code reused**: News aggregator (commits `c789d5c`, `0f22839`, `95f5fe8`)

### 6.6 Fundamentals component (**NEW INTEGRATION**)

**Input**: Yahoo Finance data — P/E, debt/equity, EPS surprise (most recent), earnings date (next)
**Output**: `fundamentals_score: Optional[float]` in [-1.0, +1.0] or `None`
**Logic**:
- **Earnings calendar veto**: if next earnings is within 24h, veto entirely (returns score=None + veto_reason)
- P/E vs sector median: above 2x median → -0.3, below median → +0.2
- Debt/equity > 2 → -0.3
- Most recent EPS surprise > 5% beat → +0.4
- Stale data (>24h): degrade weight in synthesis but not veto

**New code**: `analysis/fundamentals.py` (~200 LOC), `tests/analysis/test_fundamentals.py`

**Library**: `yfinance` (already in requirements per commit `a47d83c`)

**Rate limit**: cache per-symbol fundamentals for 4 hours; refresh on bot startup + every 4h cycle

### 6.7 Liquidity component (**NEW**)

**Input**: 20-day average daily $-volume (price × volume)
**Output**: `liquidity_score: float` in [0.0, 1.0]
**Logic**:
- Veto if 20-day avg $-vol < $5M (microcap territory)
- Scaled: $5M → 0.0, $50M → 0.5, $500M → 1.0 (log-linear)
- Cached per-symbol, refreshed daily

**New code**: pull from existing minute_bars Postgres table; small SQL query

### 6.8 Synthesis

```python
def _synthesize_view(
    symbol: str,
    trend_score: float,
    momentum_score: float,
    volatility_quality: float,
    sector_score: float,
    news_score: float,
    fundamentals_score: Optional[float],
    liquidity_score: float,
    veto_reasons: tuple[str, ...],
) -> CompositeView:
    """Compose the seven components into direction, strength, tradability.

    Direction comes from a weighted vote across signal components
    (trend, momentum, sector, news, fundamentals). Vetoes immediately
    set tradability=0 regardless of other scores.

    Strength is the magnitude of the directional consensus — high when
    components agree, low when they disagree.

    Tradability is min(volatility_quality, liquidity_score, ...) —
    weakest-link logic. One bad component kills the trade.
    """
    if veto_reasons:
        return _vetoed_view(symbol, veto_reasons, ...)

    # Direction vote: each component contributes its sign
    signed_scores = [trend_score, momentum_score, sector_score, news_score]
    if fundamentals_score is not None:
        signed_scores.append(fundamentals_score)

    signed_avg = sum(signed_scores) / len(signed_scores)
    if signed_avg > 0.2:
        direction = 'bull'
    elif signed_avg < -0.2:
        direction = 'bear'
    else:
        direction = 'neutral'

    # Strength: agreement among components
    abs_scores = [abs(s) for s in signed_scores]
    strength = sum(abs_scores) / len(abs_scores)

    # Tradability: weakest link
    tradability = min(volatility_quality, liquidity_score)

    return CompositeView(...)
```

---

## 7. Public API

### 7.1 Module location

`core/composite_view.py`

### 7.2 Function signatures

```python
def read_composite_view(
    symbol: str,
    *,
    indicators: dict,
    market_indices: "MarketIndicesCache",
    news_state: "NewsAggregator",
    fundamentals_cache: "FundamentalsCache",
    liquidity_cache: "LiquidityCache",
    force_refresh: bool = False,
) -> CompositeView:
    """Compute (or return cached) composite view for a symbol.

    Cached for 60s by default (overridable via force_refresh).
    Pure function except for cache mutations. Safe to call from
    the analysis loop on every bar.

    Failure mode: if any component raises, that component's contribution
    is set to 0.0 and a warning is logged. The view is still returned
    with reduced confidence (smaller strength).
    """
    ...


def invalidate_view_cache(symbol: Optional[str] = None) -> None:
    """Flush cached view(s). Call on operator command or symbol-level events
    (close, manual intervention, etc.). Pass None to flush all."""
    ...
```

### 7.3 Caching policy

| Component | Cache TTL | Refresh trigger |
|---|---|---|
| Trend | 30s | Per-bar (new bar arrival) |
| Momentum | 30s | Per-bar |
| Volatility quality | 5 min | Periodic, plus on `invalid_indicators` event |
| Sector | 5 min | MarketContext cache TTL |
| News | 5 min | News aggregator cache TTL |
| Fundamentals | 4 hours | Bot startup + 4h schedule + earnings date approach |
| Liquidity | 24 hours | Daily refresh from Postgres |
| **Synthesis** | **60s** | When ANY input invalidates |

---

## 8. Integration points

### 8.1 Strategies become consumers

Every strategy's `generate_signal_with_commentary()` gets a new pre-check:

```python
async def generate_signal_with_commentary(self, market_data) -> Optional[TradingSignal]:
    from core.composite_view import read_composite_view
    from core.config import Config

    if Config().ENABLE_COMPOSITE_VIEW_GATE:
        view = read_composite_view(market_data.symbol, ...)
        if not view.allows_long() and self.is_long_strategy():
            self._log_decision(market_data, "skip", "view_blocks_long",
                               direction=view.direction,
                               strength=round(view.strength, 2),
                               tradability=round(view.tradability, 2),
                               vetoes=','.join(view.veto_reasons))
            return None
        # ... same check for SHORT

    # ... existing strategy logic unchanged
```

**Important**: the check is *first* in the function, before any other logic. Cheap to skip — `read_composite_view` is cached. Adds <5ms per call when cached.

### 8.2 Risk manager observes the view

The risk manager's sizing function reads the view's strength and adjusts the conviction multiplier:

```python
# Existing market_context_conviction multiplier
mult = signal.reasoning.get('market_context_conviction', 1.0)

# NEW: view strength as additional multiplier
view = read_composite_view(signal.symbol, ...)
view_strength_mult = 0.5 + view.strength  # 0.5 to 1.5 range
mult *= view_strength_mult
```

### 8.3 Dashboard widget

Top-N "actionable" names tile: shows the 5 highest-`tradability` names with their direction + strength + top 2 bull/bear factors. Updates every 10s. Pure read-only.

### 8.4 Audit log integration

Every `read_composite_view` call writes a structured `engine_decision`:

```
component=composite_view symbol=SMCI action=computed direction=bear
  strength=0.72 tradability=0.85 trend=-0.6 momentum=-0.4 sector=+0.3
  news=+0.1 fundamentals=-0.3 liquidity=0.9 bull_factors=sector_strength
  bear_factors=trend_below_sma50,macd_below_signal,fundamentals_high_pe
```

Same log pipeline as existing audit events. Searchable via the existing dashboards.

---

## 9. Failure modes and resilience

| Failure mode | Detection | Mitigation |
|---|---|---|
| Yahoo Finance API down/slow | Timeout on `yfinance` call | Use cached fundamentals; if no cache, set `fundamentals_score = None` and proceed |
| News aggregator empty for symbol | `news_count == 0` | `news_score = 0.0` (neutral), `news_factors = ['no_news_data']` |
| Indicator NaN (insufficient bars) | `np.isnan(value)` | Veto with reason `insufficient_data`, `tradability = 0` |
| MarketIndicesCache stale | `age_sec > 600` | Use stale data but log warning; degrade `sector_score` weight by 0.5 |
| Postgres connection lost | Exception on liquidity query | Use cached liquidity; if no cache, `liquidity_score = 0.5` (neutral) and log |
| All components fail simultaneously | All inputs failed | Return `neutral` direction, `tradability = 0`, `veto_reasons = ['all_components_failed']` |
| Cache corruption | Pickle/JSON deserialize fails | Discard cache entry, recompute |
| Race condition on cache | Concurrent reads/writes | Per-symbol lock (asyncio.Lock); cache is keyed by symbol |
| Symbol mapping unknown | symbol → sector mapping miss | `sector_score = 0.0` (neutral), log gap for future mapping addition |

**Design principle**: any single component can fail without breaking the view. The view is still returned, just with that component's contribution zeroed out. The operator sees the gap via the audit log.

---

## 10. Performance budget

| Operation | Target | Worst case |
|---|---|---|
| `read_composite_view()` cached hit | <1 ms | 5 ms |
| `read_composite_view()` cold (all components run) | <50 ms | 200 ms |
| Yahoo Finance call (single symbol) | <500 ms | 5 sec (timeout) |
| News aggregator query | <100 ms | 1 sec |
| Postgres liquidity query | <50 ms | 500 ms |
| Per-bar gate check (cached) | <5 ms | 20 ms |
| Per-bar gate check (cold) | <50 ms | 200 ms |

**Per-analysis-cycle budget**: 30 symbols × <5ms (cached) = 150ms of view-gate overhead per cycle.
**Cold-start budget**: 30 symbols × <50ms = 1500ms (acceptable for one-time startup).

Cache invalidation strategy ensures cold-start is rare after first 60s.

---

## 11. Security

### 11.1 New dependencies

- **`yfinance`**: already in dependencies (commit `a47d83c`). No new security review needed.
- No new credentials. Yahoo Finance is unauthenticated public data.

### 11.2 Rate limiting

- Yahoo Finance: max 1 call per symbol per 4 hours.
- Aggregate: max 50 Yahoo calls per hour (well under the unofficial limit).
- Postgres liquidity: cached 24h, ~30 queries per day total.

### 11.3 Data handling

- Composite View data is read-only and ephemeral (cache only).
- No PII or financial credentials in any view.
- Logs are structured but don't include account balance or position sizes.

### 11.4 Threat model

Threats out of scope (already addressed by existing security work):
- API key theft (handled by `.env` + `.gitignore`)
- WebSocket auth (handled by Bearer token middleware)
- Command injection (audited in commit `07cf267`)

Threats specific to this layer:
- **Yahoo Finance poisoning**: unlikely; Yahoo is the authoritative source we'd be comparing against anyway.
- **Cache poisoning**: per-symbol locks prevent it; cache is in-process only.
- **DOS via excessive symbol watchlist**: bounded by existing `max_watchlist_size = 50`.

---

## 12. Testing strategy

### 12.1 Layers

1. **Unit tests per component**: pure-function tests with mocked inputs.
   - Coverage target: 90% per component.
   - Property tests: direction ∈ {bull, bear, neutral}, scores in valid ranges.
   - Edge cases: NaN, missing data, extreme values.

2. **Synthesis tests**: composed scenarios.
   - "Bull trend + bear news + neutral sector = ?" type tests.
   - Veto cascade: one veto wins regardless of other strong signals.

3. **Integration tests**: end-to-end with real (but mocked-at-boundary) inputs.
   - Mocked Schwab tick, real indicator computation, mocked Yahoo response.
   - Verifies the gate skip path emits the correct audit event.

4. **Backtest comparison**: with-gate vs without-gate over 60 days.
   - Same data, same strategies, only the gate differs.
   - Expected: gate-on shows fewer trades, higher per-trade EV.
   - **Decision criterion**: gate-on PF >= without-gate PF × 1.2.

5. **Sim soak**: 5 trading sessions of live data, sim-mode only.
   - Verifies gate decisions match operator's mental model.
   - Catches any per-symbol misclassifications.

### 12.2 Test file locations

- `tests/core/test_composite_view.py` — synthesis + caching + public API
- `tests/core/composite_view/test_trend_component.py` — per-component
- `tests/core/composite_view/test_momentum_component.py`
- `tests/core/composite_view/test_volatility_quality_component.py`
- `tests/core/composite_view/test_sector_component.py`
- `tests/core/composite_view/test_news_component.py`
- `tests/core/composite_view/test_fundamentals_component.py`
- `tests/core/composite_view/test_liquidity_component.py`
- `tests/integration/test_composite_view_gate.py` — strategy integration

### 12.3 Coverage targets

- Per-component: 90%
- Synthesis: 95%
- Public API: 100%
- Integration: 80% (some paths require live data)

---

## 13. Observability

### 13.1 Structured audit events

Every view computation: `engine_decision component=composite_view symbol=X action=computed ...`
Every gate decision: `engine_decision component=composite_view_gate symbol=X action=skip reason=view_blocks_long ...`

### 13.2 Metrics

Counters (per-symbol, per-day):
- `composite_view.computed` — total view computations
- `composite_view.cached_hit` — cache hits
- `composite_view.cache_miss` — cold computes
- `composite_view.component_failure.{component}` — per-component failure rate
- `composite_view.veto.{reason}` — veto counts by reason

Histograms:
- `composite_view.compute_duration_ms` — p50/p95/p99 of cold compute
- `composite_view.strength_distribution` — to validate threshold tuning
- `composite_view.tradability_distribution` — same

### 13.3 Dashboard

New widget: "Composite View Snapshot"
- Top 5 actionable names: symbol, direction (icon), strength bar, tradability
- Bottom 5 vetoed names: symbol, primary veto reason
- Updates every 10 sec
- Click for component breakdown

### 13.4 Health checks

Endpoint: `GET /api/composite-view/health` returns:
- Cache hit rate
- Component failure rate (per component, last 1h)
- Median compute time
- Number of currently-cached views

---

## 14. Migration plan

### Phase 1 — Foundation (Week 1)

**Goal**: Composite View Service exists, computes correctly, but is NOT yet a gate.

- [ ] Create `core/composite_view.py` with dataclass + `read_composite_view()` signature
- [ ] Implement 6 of 7 components (skip fundamentals initially)
- [ ] Implement synthesis with weighted-fallback logic
- [ ] Implement caching layer
- [ ] Write 90% unit test coverage per component
- [ ] Wire as **read-only logger** in the analysis loop: every bar, compute view for every watchlist symbol, log to audit event, but no strategies consume yet
- [ ] Operator reviews 1-2 days of audit logs to validate view classifications match intuition

**Ship criteria**: views computed continuously, no strategy behavior change, audit log readable.

### Phase 2 — Gate wiring (Week 2)

**Goal**: Gate the strategies behind the view, but only in sim mode initially.

- [ ] Add `Config.ENABLE_COMPOSITE_VIEW_GATE` (default False)
- [ ] Modify each strategy's `generate_signal_with_commentary()` to read view and skip if gate blocks
- [ ] Add `Config.COMPOSITE_VIEW_TRADABILITY_FLOOR` and `Config.COMPOSITE_VIEW_STRENGTH_FLOOR` (default 0.4 and 0.5)
- [ ] Flip ENABLE_COMPOSITE_VIEW_GATE=True in `Config().yaml` for sim mode
- [ ] Sim-mode soak: 5 sessions, no live impact
- [ ] Backtest 60-day comparison: with-gate vs without-gate

**Ship criteria**: backtest shows gate-on PF ≥ no-gate PF × 1.2, sim soak shows operator agrees with skip decisions ≥80% of the time.

### Phase 3 — Fundamentals integration (Week 2-3)

**Goal**: Add the 7th component (fundamentals) with Yahoo Finance integration.

- [ ] Implement `analysis/fundamentals.py` with `yfinance` calls
- [ ] Add 4h cache + earnings calendar awareness
- [ ] Unit tests + integration tests
- [ ] Wire into Composite View synthesis
- [ ] Re-run backtest

**Ship criteria**: fundamentals don't degrade overall PF; earnings veto demonstrably blocks at least one pre-earnings trade in test data.

### Phase 4 — Live enablement (Week 3-4)

**Goal**: Live trading runs through the view gate.

- [ ] Flip gate to live in `Config().yaml`
- [ ] Watch first day intensively
- [ ] Compare live decisions to operator's mental model

**Ship criteria**: 5 live sessions show ≥80% operator agreement with gate decisions, no catastrophic failures.

### Phase 5 — Breakout re-enable (Week 4)

**Goal**: With Composite View providing context, the previously-disabled breakout strategy can be safely re-enabled.

- [ ] Flip `ENABLE_BREAKOUT_LONG=True`
- [ ] Sim soak 5 sessions to confirm view correctly gates breakout entries
- [ ] Live enable

---

## 15. Risks and mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| View misclassifies a symbol → bad trade | Medium | Medium ($100-500 per occurrence) | A/B backtest, sim soak, operator override always available |
| Yahoo Finance API breaking changes | Low | Medium (fundamentals component degraded) | Cache + fallback (return None, skip component) |
| Performance regression from view computation overhead | Low | Low (50ms × 30 symbols = 1.5s cold start) | Per-symbol caching; budgeted in Section 10 |
| Cache invalidation bugs lead to stale view used for live decisions | Medium | Low-Medium | TTL strict; tests cover invalidation; audit log shows view timestamp |
| Operator disagrees with view classifications | Medium | Low | Override always works; collect disagreements as training data for future tuning |
| Component scores not actually informative | Low | High (gate adds friction without value) | Backtest comparison REQUIRED before live; if no lift, don't ship |
| New dependency (`yfinance`) breaks | Low | Low (already in deps) | Already integrated for sim mode |
| Operator burnout from extended migration | Medium | Medium | Phased rollout; each phase ships standalone value |
| Composite View becomes the bottleneck for new features | Low | High (architectural lock-in) | Components are independent; new ones add without modifying existing |

---

## 16. Alternatives considered

### 16.1 Distributed per-strategy gates

**Approach**: Add MarketContext / Direction Reader checks individually to every strategy's `generate_signal_with_commentary()`.

**Rejected because**:
- Already partially done — MarketContext is wired into 3 strategies. Today's pain shows this is insufficient.
- No central place to add new context dimensions (fundamentals, news).
- Hard to A/B test; each strategy mutates independently.
- Duplicates logic across strategies.

### 16.2 Multi-stage filter pipeline

**Approach**: Strategies fire, then signals pass through a stack of N filters (existing pattern, extended).

**Rejected because**:
- Reactive, not proactive. Strategies still waste cycles on bad symbols.
- Filters are unaware of each other; adding new ones requires manual ordering.
- No "this symbol is actionable today" concept; only "this signal passes."

### 16.3 ML predictor as the gate

**Approach**: Train a classifier that ingests all inputs and outputs go/no-go.

**Rejected because**:
- Opaque. Operator can't debug why a name was vetoed.
- Requires training data that doesn't exist yet (5-10 sessions of labeled outcomes).
- ML meta-labeling (existing) is already a downstream confidence layer; adding upstream too would compound opacity.
- Symbol-level interpretability is critical for operator trust.

### 16.4 Per-symbol mini-strategies

**Approach**: Each symbol gets its own custom strategy (Tesla strategy, Nvidia strategy, etc.).

**Rejected because**:
- Doesn't scale; every new symbol needs new code.
- Each strategy is a snowflake; can't compare or generalize.
- Operator's mental model is "view + strategy palette," not "custom code per symbol."

### 16.5 Selected: Composite View Service

**Chosen because**:
- Transparent (every component score visible).
- Composable (existing components plug in directly).
- Testable (pure functions).
- Reversible (config flag).
- Extensible (add components without touching existing code).
- Matches operator's articulated mental model.

---

## 17. Open questions

1. **Where to host fundamentals data after Yahoo Finance?** If Yahoo's free tier becomes unreliable, consider Alpha Vantage (free tier, lower rate limit), Polygon ($), or Tiingo ($). For now: Yahoo is sufficient. Revisit at Phase 4.

2. **Should the view recommend a strategy?** I.e., should the view say "this is a breakout setup" or just "direction bull strength 0.7, you figure out which strategy applies"? **Current design**: view is strategy-agnostic; strategies decide if they apply. Lets us add strategies without changing the view.

3. **What if multiple strategies want to fire on the same symbol same bar?** Already handled by correlation_guard and signal_router. Composite View doesn't change this.

4. **How does the view handle the operator's manual trades?** Composite View is read-only and doesn't manage positions. Manual trades remain unmanaged-by-bot. View can still compute on those symbols for operator's reference but doesn't trade them.

5. **Should the view be persisted to Postgres for replay/backtest?** **YES for backtest**, NO for live. Backtest replay needs the view at each historical bar. Live can compute on demand.

6. **Should view computations be ML-trained or rule-based?** **Phase 1-4 rule-based** for interpretability. After 3 months of live data, consider replacing component implementations with ML while keeping the dataclass interface stable.

7. **What's the right cache TTL for the synthesis?** 60s is a starting point. Tune based on Phase 1 audit-log review.

8. **Should the view degrade gracefully if 4+ components fail?** Currently set: any veto wins, otherwise scores zeroed. Open question: at what number of component failures does the view become unreliable enough to return tradability=0 globally? Plausibly: >2 component failures = synthesis tradability max 0.3.

---

## 18. Glossary

| Term | Definition |
|---|---|
| **Composite View** | Per-symbol holistic assessment combining 7 inputs into a single direction/strength/tradability triple. |
| **Direction** | Discrete classification: `bull`, `bear`, or `neutral`. Defaults to `neutral` when components disagree. |
| **Strength** | Conviction in the direction call, [0.0, 1.0]. High when components agree, low when split. |
| **Tradability** | Global "should we consider this symbol at all" score, [0.0, 1.0]. Driven by weakest input (volatility quality, liquidity). |
| **Component** | One of the 7 inputs feeding the view: Trend, Momentum, Volatility Quality, Sector, News, Fundamentals, Liquidity. |
| **Veto** | A hard rejection that immediately sets tradability=0 regardless of other scores. Examples: earnings within 24h, bb_lower negative, ATR > 5% of price. |
| **View Gate** | The integration point where strategies check the view before firing signals. |
| **Strategy-first** | Current architecture: strategies generate independent signals, filters reject some. |
| **Symbol-first** | Proposed architecture: per-symbol view computed first, strategies fire only when view supports them. |
| **Falling knife** | Pattern: oversold-looking buy candidate that's actually in a confirmed downtrend. Existing LONG-side filter. |
| **Rising peak** | Symmetric pattern on SHORT side. Existing v-rising-peak-filter-2026-06-08 (currently inert due to structural mismatch). |
| **MarketContext** | Existing service computing regime/sector/conviction at portfolio level. Becomes an input to Composite View. |
| **Direction Reader** | Existing 5-component price-direction algorithm. Becomes an input to Composite View. |
| **PAVS-killer** | Colloquial term for the volatility-quality veto that rejects degraded symbols (negative bb_lower, extreme ATR). Named after PAVS (2026-06-09 incident: +49% intraday rip would have catastrophically stopped any short). |

---

## Appendix A — File inventory

New files:
- `core/composite_view.py` — main service (~400 LOC)
- `analysis/fundamentals.py` — Yahoo Finance integration (~200 LOC)
- `tests/core/test_composite_view.py` — synthesis + caching (~300 LOC)
- `tests/core/composite_view/test_*_component.py` — 7 component test files (~150 LOC each)
- `tests/analysis/test_fundamentals.py` — fundamentals integration tests (~150 LOC)
- `tests/integration/test_composite_view_gate.py` — strategy integration (~200 LOC)
- `docs/COMPOSITE_VIEW_ARCHITECTURE.md` — this document

Modified files:
- `core/config.py` — 3 new properties (`ENABLE_COMPOSITE_VIEW_GATE`, `COMPOSITE_VIEW_TRADABILITY_FLOOR`, `COMPOSITE_VIEW_STRENGTH_FLOOR`)
- `strategies/builtin.py` — view-gate pre-check in each `generate_signal_with_commentary()`
- `strategies/news_strategy.py` — same view-gate pre-check
- `risk/manager.py` — multiply Kelly sizing by view strength
- `api/routes.py` — new `/api/composite-view/health` endpoint, dashboard widget data
- `Config().yaml` — new flags + thresholds (operator-managed, gitignored)

Estimated total new code: ~2000 LOC source + ~2000 LOC tests = 4000 LOC. Sustainable for one developer over 3-4 weeks.

---

## Appendix B — Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-06-09 | Adopt symbol-first architecture | Empirical: 2026-06-09 trading session showed strategy-first hits ceiling. Operator articulated vision explicitly. |
| 2026-06-09 | Use rule-based components, not ML | Interpretability is critical for operator trust; ML can replace later if needed. |
| 2026-06-09 | Yahoo Finance for fundamentals (Phase 3) | Already in dependencies; free tier sufficient for current symbol count. |
| 2026-06-09 | Phased rollout (5 phases over 4 weeks) | Each phase ships standalone value; reversible at any stage. |
| 2026-06-09 | Discrete direction (bull/bear/neutral) over continuous | Forces a clean go/no-go decision; avoids "kinda bull" trades. |

---

*End of document. Total: ~600 lines. Implementation tracked under tasks #38–#45 (to be created).*
