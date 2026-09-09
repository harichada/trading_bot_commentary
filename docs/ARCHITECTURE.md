# System Architecture

*Production trading system architecture. Updated 2026-09-08.*

---

## 1. System Overview

The trading bot is an event-driven system that:
1. Streams real-time market data from Schwab
2. Aggregates news sentiment from multiple free sources
3. Evaluates trading signals via multiple strategies
4. Manages positions with bracket orders and FSM-based exits
5. Explains every decision via real-time commentary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Trading Engine Core                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │ Position Loop │   │ Analysis Loop │   │ Screener Loop │                  │
│  │    (1s)       │   │   (30s RTH)   │   │    (120s)     │                  │
│  │               │   │   (300s off)  │   │               │                  │
│  │  FSM exits    │   │  Signals +    │   │  Watchlist    │                  │
│  │  Stop/Trail   │   │  ML scoring   │   │  refresh      │                  │
│  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘                  │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                              │
│                    ┌─────────────────┐                                      │
│                    │   TaskSupervisor │ ◄── Crash recovery, circuit-break   │
│                    └─────────────────┘                                      │
│                                                                             │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │  News Loop    │   │ Market Index  │   │    State      │                  │
│  │   (20s)       │   │    Loop       │   │  Persistence  │                  │
│  │               │   │               │   │    (30s)      │                  │
│  │  NewsBus      │   │  SPY/VIX/     │   │               │                  │
│  │  refresh      │   │  sector ETFs  │   │  JSON save    │                  │
│  └───────┬───────┘   └───────────────┘   └───────────────┘                  │
│          │                                                                  │
│          ▼                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                           NewsBus                                      │  │
│  │   symbol → [ScoredNewsItem(headline, sentiment, source, age_sec)]     │  │
│  │   Thread-safe, async-safe, TTL-based eviction                          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Data Flow                                       │
│                                                                             │
│  Schwab Stream ──► QuoteCache ──► PriceBook ──► Position FSM               │
│                                       │                                      │
│  Yahoo/Google/MW ──► NewsBus ──► Strategies ──► Signal Router ──► Orders   │
│                                       │                                      │
│                              ML Ensemble ──┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Supervised Task Model

All long-running work is managed by `TaskSupervisor`:

| Task | Cadence | Priority | Purpose |
|------|---------|----------|---------|
| `position_loop` | 1s | CRITICAL | FSM exit dispatcher — stops, trails, breakeven |
| `analysis_loop` | 30s RTH / 300s off-hours | NORMAL | Strategy signals, ML scoring, entry routing |
| `screener_loop` | 120s | NORMAL | Dynamic watchlist refresh from Yahoo screener |
| `news_loop` | 20s | NORMAL | Refresh NewsBus from FreeNewsAggregator |
| `schwab_stream` | continuous | NORMAL | Real-time tick stream via WebSocket |
| `quote_streamer` | fallback polling | NORMAL | REST quote fallback when stream unhealthy |
| `market_indices` | 60s | NORMAL | SPY/VIX/sector ETF regime strip |
| `state_persistence` | 30s | NORMAL | Write trading_state.json |

**Crash behavior:**
- NORMAL tasks: 5s → 30s → 60s backoff, circuit-break after 3 crashes in 300s
- CRITICAL tasks: 2s → 5s → 15s backoff, same breaker threshold
- Circuit-broken tasks require operator reset via `POST /api/supervisor/reset/{name}`

---

## 3. NewsBus Contract

The `NewsBus` is the single source of truth for news sentiment. All strategies
read from it; none fetch independently.

### Data Model

```python
@dataclass
class ScoredNewsItem:
    id: str                    # hash of url
    symbol: str
    headline: str
    summary: str
    source: str                # "Yahoo Finance", "Google News", "MarketWatch"
    source_tier: int           # 1=primary (Yahoo), 2=secondary (Google), 3=scrape
    url: str
    published_time: datetime
    fetched_at: datetime       # when we retrieved it
    sentiment_score: float     # VADER compound, -1 to +1
    sentiment_confidence: float
    impact: NewsImpact         # LOW, MEDIUM, HIGH (keyword detection)

    def age_sec(self) -> float:
        """Seconds since publication."""
        return (datetime.now() - self.published_time).total_seconds()
```

### Bus API

```python
class NewsBus:
    async def get_items(self, symbol: str, max_age_sec: float = 14400) -> list[ScoredNewsItem]
    async def get_freshest(self, symbol: str) -> Optional[ScoredNewsItem]
    def has_high_impact(self, symbol: str, since_sec: float = 300) -> bool
    def get_symbols_with_high_impact(self, since_sec: float = 300) -> set[str]
```

### Guarantees

1. **Freshness**: Items older than `ttl_sec` (default 4h) are evicted
2. **Thread-safety**: All reads/writes protected by asyncio.Lock
3. **Age transparency**: Every item carries `fetched_at`; strategies see true age
4. **Single fetch path**: `news_loop` is the sole writer; strategies never fetch directly

### Observability

Every strategy decision logs:
- `news_age_sec`: age of the freshest item consulted
- `news_count`: number of items in the bus for that symbol
- `news_source`: source of the freshest item

### News Thesis Gates (v-newsbus-gates-2026-09-09)

Deterministic sizing based on news quality:

| Gate | Condition | Action |
|------|-----------|--------|
| **VETO_STALE** | Freshest news > `NEWS_GATE_MAX_AGE_SEC` (30 min) | Block trade |
| **VETO_LOW_TIER** | All sources below `NEWS_GATE_SOURCE_TIER_FLOOR` | Block trade |
| **VETO_NO_CORROBORATION** | Zero fresh articles | Block trade |
| **REDUCED_SIZE** | Single source, fresh | 0.5× position size |
| **FULL_SIZE** | Multi-source fresh OR high-impact single | 1.0× position size |

Source tiers:
- Tier 1: Yahoo Finance (API-backed, curated)
- Tier 2: Google News (aggregated)
- Tier 3: MarketWatch scrape (lowest reliability)

Gate counters are tracked in `NewsBusStats` for observability.

---

## 4. News Verifier Promotion Criteria

The `ENABLE_NEWS_VERIFIER` flag gates the fresh-news re-verification system.
It is **disabled by default** (`False`) and should only be enabled after meeting
research-validated promotion criteria.

### Stage A: Initial Validation

**Criteria:**
- Minimum 80 verified signals in walk-forward validation
- Profit Factor (PF) ≥ 1.30 on gated signals
- No significant degradation in signal quality vs ungated baseline

**Actions:**
- Run `research/news_strategy_validation.py` with verifier enabled
- Compare gated vs ungated signal outcomes
- Track in `bot_shadow_news_vetoes` table

### Stage B: Production Promotion

**Criteria:**
- Minimum 200 verified signals
- Profit Factor (PF) ≥ 1.50 on gated signals
- Win rate stable at 45-55%
- Veto accuracy ≥ 60% (vetoed signals would have lost)

**Actions:**
- Set `ENABLE_NEWS_VERIFIER: true` in Config.yaml
- Monitor via dashboard news verifier metrics
- Maintain `NEWS_VERIFIER_ADVISORY: true` for first week post-promotion

### Current Status

```yaml
# Config.yaml — DO NOT CHANGE unless Stage B criteria met
trading:
  enable_news_verifier: false        # Disabled until n≥200, PF≥1.50
  news_verifier_advisory: true       # Logging verdicts for Stage A data collection
```

### Metrics Collection

While disabled, advisory mode logs every verifier verdict:
```
engine_decision component=news_verifier_advisory action=advisory 
  reason=<verdict> fresh_count=N latest_age_min=...
```

Use `scripts/analyze_news_vetoes.py` (future) to compute promotion metrics.

---

## 5. Autonomy Profiles

The bot supports two operating profiles, selected via config or environment:

### Supervised (default)

```yaml
profile: supervised  # or omit for default

order_management:
  require_close_confirmation: true
  confirmation_timeout_sec: 30
  confirmation_timeout_action: deny  # block the close if UI doesn't respond
```

- Every exit (stop, trail, proactive) requests UI confirmation
- 30s timeout → close is **denied** (position stays open)
- Operator must manually intervene if UI is unavailable

### Autonomous Live

```yaml
profile: autonomous_live

order_management:
  require_close_confirmation: false
  # OR, for monitored autonomy:
  require_close_confirmation: true
  confirmation_timeout_sec: 30
  confirmation_timeout_action: execute  # fail-open: execute the close

trading:
  max_daily_loss_circuit: 0.02      # 2% of equity
  flatten_on_circuit: true          # close all positions when circuit trips
  
# Risk controls remain active — the bot will NEVER:
#   - Exceed max_positions (10)
#   - Exceed max_position_value ($25k)
#   - Enter outside regular trading hours
#   - Trade when quote stream is stale
```

**Key difference**: `confirmation_timeout_action: execute` means the bot will
close the position if the UI doesn't respond within the timeout. This prevents
the scenario where a losing position stays open because the operator wasn't
watching the dashboard.

### Profile Selection

1. Environment variable: `TRADING_PROFILE=autonomous_live`
2. Config file: `profile: autonomous_live` at root level
3. API: `PUT /api/settings/profile` (runtime toggle)

---

## 6. Economic Calendar Blackouts

The bot pauses new entries during high-impact economic events (CPI, FOMC, etc.)
via the `EconCalendarProvider` interface.

### Provider Hierarchy

1. **ApiEconCalendar**: Uses external API (Trading Economics, etc.) if `ECON_CALENDAR_API_URL` is set
2. **ConfigEconCalendar**: Reads events from `Config.yaml` under `trading.econ_calendar_events`
3. **DatedEconCalendar** (default): Uses actual event dates (v-econ-calendar-dated-2026-09-09)

### CRITICAL: Dated vs Static Calendar (v-econ-calendar-dated-2026-09-09)

**Problem fixed**: The original `StaticEconCalendar` treated EVERY Wednesday 14:00
as FOMC and 14:30 as Fed speech using `days_of_week` patterns. This caused phantom
blackouts on non-FOMC Wednesdays, freezing LIVE mode analysis while SIM ignored them.

**Solution**: `DatedEconCalendar` is now the default. It uses actual event dates
(e.g. 2026-09-17 for September FOMC) instead of weekday patterns. FOMC only happens
8 times per year, not every Wednesday.

**Static patterns are FORBIDDEN** for FOMC/Fed events. Use explicit dates via:
- `DatedEconCalendar` (default, has 2026-2027 FOMC dates built-in)
- `ConfigEconCalendar` with `date: "YYYY-MM-DD"` field per event

### Blackout Behavior: LIVE vs SIM

Both LIVE and SIM modes now handle blackouts consistently (v-econ-calendar-dated-2026-09-09):

| Aspect | Old Behavior | New Behavior |
|--------|--------------|--------------|
| Analysis during blackout | LIVE: **skipped entirely** | LIVE: **continues** |
| | SIM: ran normally | SIM: continues |
| New entries during blackout | LIVE: blocked (by skip) | LIVE: **soft veto at signal router** |
| | SIM: allowed | SIM: soft veto at signal router |
| `strategy_decision` logs | LIVE: none during blackout | LIVE: **logged** (veto reason shown) |
| | SIM: logged | SIM: logged |

**Key principle**: Blackout blocks **new entries only**, not analysis. The bot
continues to think, evaluate opportunities, and log decisions. It just doesn't
place new orders during high-volatility news windows. This matches how a pro
trading desk operates.

### Observability

When a blackout is active, the following are logged:

1. **INFO-level log** in `_evaluate_trading_conditions`:
   ```
   econ_blackout_in_effect event=FOMC ends=14:30 ET remaining_min=15.0 — analysis continues, new entries blocked
   ```

2. **Commentary** visible in dashboard:
   ```
   📰 Econ Blackout Active — FOMC Rate Decision
   Economic event blackout in effect until 14:30 ET (~15min remaining).
   Analysis continues; new entries blocked at signal router.
   ```

3. **Audit log** when signal is vetoed:
   ```
   engine_decision component=econ_blackout action=skip reason=blackout_soft_veto event=FOMC ends=14:30
   ```

### Configuration

```yaml
trading:
  econ_calendar_events:
    - name: "CPI Release"
      type: cpi
      time: "08:30"
      duration: 15
      date: "2026-09-10"  # Explicit date (required for accurate blackouts)
    - name: "FOMC Decision"
      type: fomc
      time: "14:00"
      duration: 30
      date: "2026-09-17"  # Explicit FOMC date (8 per year)
    - name: "Powell Speech"
      type: fed_speech
      time: "14:30"
      duration: 45
      date: "2026-09-17"  # Same day as FOMC
```

### Default Blackout Windows (DatedEconCalendar)

FOMC 2026 dates (announcement days at 14:00 ET):
- Jan 29, Mar 19, May 7, Jun 18, Jul 30, Sep 17, Nov 5, Dec 17

CPI releases (~10th-13th of each month at 08:30 ET):
- Jan 14, Feb 12, Mar 11, Apr 10, May 13, Jun 10, Jul 14, Aug 12, Sep 10, Oct 13, Nov 12, Dec 10

Jobs Report (first Friday of each month at 08:30 ET):
- Automatically calculated

### Future API Integration

Set environment variables to use a real calendar API:
```bash
ECON_CALENDAR_API_URL=https://api.tradingeconomics.com/calendar
ECON_CALENDAR_API_KEY=your_api_key
```

The API provider falls back to `DatedEconCalendar` if the API is unavailable.

---

## 7. Explicit Non-Goals

This system is designed for disciplined, measurable, rule-based trading.
The following are **explicitly out of scope**:

### No LLM Trade Placement from Headlines

The bot does **not** use large language models to interpret news headlines
and place trades. VADER sentiment scoring is the extent of NLP. Rationale:
- LLM outputs are non-deterministic and unauditable
- Backtesting LLM strategies is nearly impossible
- The edge, if any, decays as others adopt the same models

### No Model Retrain on Every Article

The ML ensemble (RandomForest/XGBoost/LightGBM) is trained offline on
historical data. It does **not** retrain in response to individual news
articles. Online learning is limited to outcome-based weight adjustments
after trades close, not real-time news ingestion.

### No Social Media Stubs Pretending to Be Live

`_EmptySocialTracker` and `_EmptyEarningsCalendar` exist in the codebase
as stubs returning empty/neutral data. These are **not** live integrations.
The bot does not consume:
- Reddit sentiment
- StockTwits mentions
- Twitter/X feeds
- Finnhub social metrics

If these sources are not wired to real APIs, they return neutral values.
The dashboard clearly marks them as "not configured."

### No Profitability Guarantees

This is a trading system, not a money printer. Historical backtests show
positive expectancy under specific market regimes, but:
- Past performance does not guarantee future results
- Regime changes can invalidate learned patterns
- The system can and will lose money

---

## 8. Success Metrics

These metrics define what "working correctly" means:

### Latency

| Metric | Target | Measurement Point |
|--------|--------|-------------------|
| Signal → Order | < 2s | `signal_created_at` to `order_placed_at` in audit log |
| News age at decision | < 30 min | `news_age_sec` field on entry signals |
| Quote staleness | < 15s | `quote_max_stale_sec` config, enforced in position_loop |

### Risk Controls

| Metric | Target | Circuit |
|--------|--------|---------|
| Max daily loss | 2% of equity | `TradingLossBreaker` trips, halts all entries |
| Max drawdown | 10% | Alert only (no auto-flatten by default) |
| Override rate | < 10% | Operator manual closes / total closes |

### Quality

| Metric | Calculation | Target |
|--------|-------------|--------|
| Expectancy | avg_win × win_rate - avg_loss × (1 - win_rate) | > 0 |
| Profit Factor | gross_wins / gross_losses | > 1.2 |
| Win Rate | wins / total_trades | 45-55% (with R:R > 1.5) |

### System Health

| Metric | Target |
|--------|--------|
| Task circuit-breaks/week | 0 |
| Quote stream disconnects/day | < 3 |
| News fetch failures/hour | < 5 |

### Observability Checklist

Every trade entry must log:
- [ ] `wake_reason` (poll, news_alert, screener_change)
- [ ] `news_age_sec` (if news strategy)
- [ ] `profile` (supervised, autonomous_live)
- [ ] `confirmation_timeout_action` (if applicable)

---

## 9. Configuration Reference

### Profile-Related Settings

```yaml
# Top-level profile selector
profile: supervised  # or autonomous_live

order_management:
  require_close_confirmation: true
  confirm_only_losses: false
  confirm_threshold_percent: 5
  confirmation_timeout_sec: 30
  confirmation_timeout_action: deny  # or execute

trading:
  max_daily_loss_circuit: 0.02
  flatten_on_circuit: false  # true for autonomous_live
  enable_proactive_exit: true
  enable_thesis_revalidation: false
  
  # Loop cadences
  position_loop_sec: 1.0
  analysis_loop_rth_sec: 30.0
  analysis_loop_off_hours_sec: 300.0
  screener_loop_sec: 120
  news_loop_sec: 20
```

### NewsBus Settings

```yaml
news:
  bus_ttl_sec: 14400          # 4 hours
  bus_max_items_per_symbol: 50
  high_impact_keywords:
    - earnings
    - beat
    - miss
    - sec
    - investigation
    - guidance
    - upgrade
    - downgrade
  source_tiers:
    yahoo_finance: 1
    google_news: 2
    marketwatch: 3
```

---

## 10. Appendix: FSM Exit States

```
                    ┌──────────────────────────────────────────┐
                    │                                          │
                    ▼                                          │
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  OPEN   │───►│ TRAILING│───►│ EXITING │───►│ CLOSED  │      │
└─────────┘    └─────────┘    └─────────┘    └─────────┘      │
     │              │              │                           │
     │              │              │         ┌─────────┐       │
     │              │              └────────►│ ZOMBIE  │───────┘
     │              │                        └─────────┘  (recovery)
     │              │
     └──────────────┴─────────────────────────────────────────►
                            (direct close paths)
```

Position states and allowed transitions are defined in `core/position_state.py`.
The FSM ensures that:
- A position in EXITING cannot have new exit rules fire
- A ZOMBIE position (close failed) gets operator attention
- State transitions are logged for audit

---

*End of architecture document.*
