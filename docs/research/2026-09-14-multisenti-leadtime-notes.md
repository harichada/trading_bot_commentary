# Multisenti / NewsBus — Lead-time & Stage A Observability Notes

**Date:** 2026-09-14 (America/New_York)  
**Scope:** Brief research for measurement design — free RSS / public headlines vs market moves; Anthropic AI-slowdown example; Stage A log fields.  
**Rule:** No fabricated stats. Soft claims marked **unverified**.

---

## 1) Typical lead-time: free RSS / public headlines vs market move

### Scheduled macro (FOMC / econ prints) — academic (stronger)

| Claim | Status | Source |
|---|---|---|
| Highly liquid index products (SPY, E-mini ES) respond to **macro announcement surprises within ~5 milliseconds**; trading intensity jumps >100× after release. Profits for low-latency demanders are modest (~$19k SPY / ~$50k ES **per event** in their sample) and did not rise as speed rose. | **Sourced** | Chordia, Green, Kottimukkalur, *Review of Financial Studies* (2018), “Rent Seeking by Low-Latency Traders…” ([DOI](https://doi.org/10.1093/rfs/hhy025)) |
| FOMC **press conferences** have become the dominant policy-news channel vs statements alone in recent years; authors recommend event windows that cover statement **and** press conference. (Event-study design; not RSS latency.) | **Sourced** | Acosta et al., FRBSF WP 2025-30, “Financial Market Effects of FOMC Communication…” ([PDF](https://www.frbsf.org/wp-content/uploads/wp2025-30.pdf); [page](https://www.frbsf.org/research-and-insights/publications/working-papers/2025/12/financial-market-effects-of-fomc-communication-evidence-from-a-new-event-study-database/)) |

**Implication for free RSS:** On scheduled FOMC/macro releases, institutional machines already reprice in **milliseconds**. A retail bot on free RSS / polled public feeds is structurally after the first move. That does **not** imply zero value for slower horizons (minutes→hours), but **beating the initial move** with free feeds is the wrong default hypothesis.

### Fed speak (ad-hoc speeches / interviews) — softer

| Claim | Status | Notes |
|---|---|---|
| Ad-hoc Fed speak is less “clock-synchronized” than a scheduled FOMC statement drop; first public path is often wire / X / live TV, then aggregators. | **Unverified (mechanism plausible)** | No peer-reviewed ms-vs-RSS comparison found in this pass. |
| Free RSS still typically lags professional wires / squawk / social for the first market-moving quote. | **Unverified (practitioner consensus)** | Aligns with vendor/practitioner anecdotes below; treat as prior, measure with Stage A timestamps. |

### Earnings releases — academic (stronger on speed; weaker on RSS)

| Claim | Status | Source |
|---|---|---|
| After-hours earnings: for liquid names, price discovery is extremely fast; in a 2016–2020 subsample, simulated profits shrink or vanish once spreads / short delays are imposed. Example path: AAPL 2020-07-30 — transactions begin ~**15 ms** after announcement, >800 trades in first second. | **Sourced** | “Warp Speed Price Moves: Jumps after Earnings Announcements” ([arXiv PDF](https://arxiv.org/pdf/2601.08962)) |
| After-hours earnings discovery is largely via **quote updates**, not trades; trade-based returns understate speed/magnitude vs midquotes. | **Sourced** | Grégoire & Martineau, *Journal of Accounting Research* (2022), “How is Earnings News Transmitted to Stock Prices?” ([Wiley](https://onlinelibrary.wiley.com/doi/10.1111/1475-679X.12394)) |

### Free RSS / public aggregator latency — practitioner (soft)

| Claim | Status | Source |
|---|---|---|
| Web-scraped / RSS news often **5–30 minutes** behind the wire; unsuitable for short-horizon production trading (OK for EOD / longer). | **Unverified (vendor blog)** | Lycore, “Financial News Sentiment Analysis…” (2026-07-22) ([link](https://www.lycore.com/blog/financial-news-sentiment-analysis/)) — no methodology or measured sample published in-page. |
| PR Newswire-style RSS reportedly updates **~5–10 minutes** after article “live” time (anecdotal). | **Unverified (forum)** | r/algotrading “News feed latency” thread ([reddit](https://www.reddit.com/r/algotrading/comments/1ozljry/news_feed_latency/)) |
| Same thread: scheduled econ/earnings alpha described as mostly gone after hundreds of ms–few seconds for wire-tier participants. | **Unverified (forum anecdote)** | Same thread; consistent *directionally* with Chordia et al., not a substitute. |
| Practitioner build note: aggregator ~15 min behind wire while stock repriced in ~1 min → lag ≫ repricing window. | **Unverified (personal blog)** | daniliuc.com “Kingfisher…” ([link](https://daniliuc.com/blog/i-built-a-trading-bot-here-is-the-boring-verifiable-version/)) |

### CEO / AI policy shocks

| Claim | Status | Notes |
|---|---|---|
| Unlike clocked FOMC/earnings, CEO/policy essays often land **off-session** (weekend / after-hours). Free RSS can still *surface* the story before cash open, but the “race” is usually **session-open / gap / first-hour drift**, not a ms wire race. | **Partially observed** | See §2 Anthropic example (weekend essay → Monday session). |
| Retail bots without paid wires **usually too late to beat the first liquid move** on scheduled hard events; for off-session narrative shocks, they may still be late vs futures/premarket + paid squawk, but Stage A should measure residual T+1m…1h edges rather than assume zero. | **Prior (skeptical default)** | Supported for scheduled macro by academic ms evidence; for narrative shocks, measure — do not invent win rates. |

### Can retail bots without paid feeds beat the move?

**Skeptical default: usually too late for the first move** on FOMC/macro/earnings when professionals have machine-readable wires + colocated stack.  
**What remains plausible to measure (not claim):** residual drift, second-wave interpretation, basket cross-asset lag, and false-positive cost of acting on stale headlines — using Stage A shadow actions + forward returns.

---

## 2) Concrete example: Anthropic CEO AI-slowdown comments → AI/semis (2026-09)

### Timeline (sourced at day resolution; tick timestamps **not** found)

| When | What | Source |
|---|---|---|
| **Sat 2026-09-12** | Anthropic CEO Dario Amodei publishes essay calling to slow the pace of frontier AI capability advances (“We Must Pace the Frontier” / related coverage). Shared on X; Altman and Musk publicly agree same weekend. | NYT ([link](https://www.nytimes.com/2026/09/12/technology/anthropic-dario-amodei-ai-slowdown.html)); Politico ([link](https://www.politico.com/news/2026/09/12/anthropic-ceo-dario-amodei-seeks-immediate-slowdown-artificial-intelligence-01073519)); BBC ([link](https://www.bbc.com/news/articles/c14dpgm0rg4o)) |
| **Sun 2026-09-13** | Amodei expands on CBS “Sunday Morning” (per NBC). | NBC ([link](https://www.nbcnews.com/business/markets/stocks-tumble-ai-leaders-warning-slowdown-ipos-amodei-altman-rcna597643)) |
| **Mon 2026-09-14** | First U.S. cash session after the weekend comments; AI/semis sell off in early trade; some indexes recover off lows later. | Reuters, NBC, CNBC, Proactive |

**Exact `source_published_ts` (essay post time UTC) and tick-level `price_move_ts`:** **not located** in this pass → mark **unverified / not found**. Do not invent ms lead-time for this event.

### Same-day / early-session moves (numbers only if sourced)

From **Reuters** (2026-09-14 Asia/markets wrap; U.S. early trading figures as reported):

- Nasdaq 100: **−1.2%** early (to a six-week low, as reported)
- Philadelphia Semiconductor Index (SOX): **−5.1%**
- **NVDA: −3.6%**
- **AMD: −5.6%**
- **MU: −6%**
- Also cited: LRCX/AMAT **−6%** each; SoftBank as much as **−13.2%** (Asia)

Source: [Reuters](https://www.reuters.com/world/china/ai-linked-asian-stocks-slump-after-top-lab-ceos-call-slowing-down-technologys-2026-09-14/)

From **NBC** (updated through afternoon ET):

- Nasdaq 100 futures: **−1.5%** early; NDX cash fell as much as **~1.8%** early
- **NVDA:** fell **about 3%** (session description)
- SOX: **plunged 5%**
- Indexes later recovered off worst levels (by ~2 p.m. ET, Nasdaq Composite nearly flat; S&P ~−0.3% as reported)

Source: [NBC](https://www.nbcnews.com/business/markets/stocks-tumble-ai-leaders-warning-slowdown-ipos-amodei-altman-rcna597643)

From **Proactive Investors** (morning ET, 2026-09-14 ~12:12 EDT stamp on page):

- **NVDA:** **−3%** at **$210.86** (morning)
- **MU:** **−6%** at **$914.30**
- **AVGO:** **−5%** at **$344.50** (also cites guidance context)

Source: [Proactive](https://www.proactiveinvestors.com/companies/news/1098509/nvidia-and-other-chipmakers-suffer-as-ai-warnings-raise-spending-concerns-1098509.html)

### SMCI

| Claim | Status |
|---|---|
| **SMCI −8%** to **$37.03** on 2026-09-14, in an AI-hardware group move; article also emphasizes **HPE downgrade** and peer sympathy (DELL −5%), while noting weekend Amodei/Altman comments as sentiment backdrop. | **Sourced number, mixed attribution** — [24/7 Wall St.](https://247wallst.com/investing/2026/09/14/hewlett-packard-enterprise-sinks-9-on-downgrade-after-triple-digit-run-dell-falls-5-super-micro-drops-8/). Do **not** treat as a clean single-cause Amodei→SMCI causal print without more data. |
| SMCI named in major Reuters/NBC ticker lists for this selloff | **Not found** in those wires’ primary ticker lists above |

### Measurement takeaway for this example

- This is a **weekend narrative → Monday open/session** event, **not** a free-RSS-vs-HFT millisecond race.
- Useful Stage A question: given `bus_ingest_ts` relative to Sunday night / Monday premarket / RTH open, what was `news_age_sec` and did shadow **exit/size/skip** improve vs always-flat on basket `{NVDA, MU, AMD, SMCI, SOX}`?
- Do **not** claim retail RSS “beat” or “missed by X ms” without measured timestamps.

---

## 3) NewsBus Stage A — observability fields to log

Minimum schema for lead-time / residual-edge measurement:

| Field | Type / notes |
|---|---|
| `event_id` | Stable hash of source URL + headline + published ts (dedupe) |
| `source_id` / `source_tier` | e.g. rss_free, wire_paid, x_official, edgar |
| `source_published_ts` | Publisher’s claimed publish time (UTC); nullable if missing |
| `bus_ingest_ts` | Wall clock when NewsBus first saw the item (UTC, monotonic companion optional) |
| `news_age_sec` | `bus_ingest_ts − source_published_ts` (null if either missing) |
| `headline` / `url` | Raw text + link (store truncated if needed) |
| `theme_tags` | Controlled vocab: `fomc`, `fed_speak`, `earnings`, `ceo_policy`, `ai_policy`, … |
| `ticker_basket` | List of symbols Stage A maps for this theme (e.g. `["NVDA","MU","AMD","SMCI"]`) |
| `action_shadow` | One of `exit` \| `size` \| `skip` (and optional size delta / reason code) |
| `fwd_ret_1m` | Basket or per-ticker forward return from decision/ref price, **T+1m** |
| `fwd_ret_5m` | **T+5m** |
| `fwd_ret_15m` | **T+15m** |
| `fwd_ret_1h` | **T+1h** |

### Recommended companion fields (still Stage A; cheap)

- `session_state`: `rth` \| `pre` \| `ah` \| `weekend`
- `ref_price_ts` / `ref_price`: price timestamp used as t0 for forward returns
- `price_feed_tier`: free delayed / SIP / etc. (so lead-time is not confounded with bad marks)
- `is_scheduled_clock_event`: bool (FOMC drop vs narrative shock)
- `duplicate_of_event_id`: first-seen winner for near-dupes

### Stage A evaluation questions these fields answer

1. Distribution of `news_age_sec` by `source_tier` and `theme_tags`.
2. Conditional forward returns given `action_shadow`, stratified by `news_age_sec` buckets and `session_state`.
3. Whether free RSS ever shows positive expectancy at T+5m/15m/1h **after** filtering scheduled clock events (where ms literature says first move is already gone).

---

## Sources (compact)

1. Chordia, Green, Kottimukkalur (2018), RFS — macro news ≤~5 ms into SPY/ES.  
2. Acosta et al., FRBSF WP 2025-30 — FOMC statement vs press conference importance.  
3. Grégoire & Martineau (2022), JAR — after-hours earnings via quotes.  
4. arXiv:2601.08962 — after-hours earnings jump speed / delay kills alpha in later subsample.  
5. Lycore blog (2026-07) — RSS/scrape **5–30 min** claim (**unverified vendor**).  
6. r/algotrading news-feed latency thread — RSS **5–10 min** anecdotes (**unverified**).  
7. Reuters / NBC / Proactive / 24/7 Wall St. (2026-09-12…14) — Amodei essay weekend; Mon AI/semis moves (NVDA/MU/SOX etc.).  

---

*End of brief.*
