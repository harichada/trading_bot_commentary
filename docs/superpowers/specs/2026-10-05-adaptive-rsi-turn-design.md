# Adaptive RSI Turn Engine — Phase 1 design (Level 1 + tape)

Status: design for approval (2026-10-05). No engine code written yet.
Scope: replaces the fixed-threshold `rsi_dip` rule with an explainable, bounded
turn score driven by the symbol's own RSI distribution, RSI momentum and
divergence, tape absorption (cumulative volume delta) and Level 1 book pressure.
Level 2 is explicitly out of scope (Phase 2).

Directive constraints honoured: sizes are no longer discarded; aggressor tagging
uses the quote rule live and the tick rule on the historical trade archive;
four explainable sub-components; bounded 0–100 score with sign-constrained,
regularised, cross-validated weights; every oversold occurrence logged with the
factor breakdown and the forward 1R/stop outcome; all rolling state is O(1)
per tick.

---

## 1. Data reality

| source | what | depth | status |
|---|---|---|---|
| Schwab stream `LEVELONE_EQUITIES` | last, bid, ask, bid_size, ask_size, last_size, total_volume, net_change | ~20 scanned names | live; sizes received but dropped at `core/schwab_stream.py` → `PriceBook.apply_tick_sync` |
| Postgres `trade_ticks` | ts, symbol, price, size, exchange, conditions (trade prints, no quotes) | 2024-10-01 → 2025-03-10, 12 names (MU, NVDA, PLTR, MSFT, AAPL, …), ~370k prints/day on MU | historical, tick-rule aggressor only |
| Postgres `minute_bars` / research cache | OHLCV 1-min → 5-min and daily | 2021 → now, 539 names | historical |
| Level 2 (`NASDAQ_BOOK`, `NYSE_BOOK`) | depth | supported by the broker library | not subscribed — Phase 2 |

Consequence: the Level 1 component can only be *fitted* forward (from the
shadow ledger); the tape and RSI components can be fitted on the five-month
archive, with the bull-regime caveat the directive names.

---

## 2. Ingestion: the microstructure book

New module `core/micro_book.py`, one `SymbolMicro` per subscribed symbol, owned
by a process-wide `MicroBook` singleton installed next to `PriceBook`. The
stream handler calls `MicroBook.instance().on_tick(sym, ts, last, bid, ask,
bid_size, ask_size, last_size, total_volume)` right after `apply_tick_sync`.
Every call is O(1), allocation-free after warm-up, and runs on the stream
task only; readers take a snapshot (a copy of the small arrays) so the event
loop never blocks on a lock.

### 2.1 Trade detection from a Level 1 stream

Schwab sends partial updates. A *trade* is recognised when `last_size` or
`total_volume` changes between consecutive updates for the symbol:

```
Δvol = total_volume_t − total_volume_{t−1}
size  = Δvol if Δvol > 0 else last_size if last_size changed else 0
```

A tick with `size == 0` is a quote-only update and feeds the quote ring only.

### 2.2 Aggressor tagging

Live (quote rule, per directive; never downgraded to the tick rule when a
quote is present):

```
side = +1  if last ≥ ask_prev          (buyer-initiated)
       −1  if last ≤ bid_prev          (seller-initiated)
       sign(last − last_prev_trade)    if bid_prev < last < ask_prev
       side_prev                       if that difference is 0 (flat inherits)
```

`bid_prev`/`ask_prev` are the quote in force *before* the trade print (the
previous update), which is what the trade hit.

Historical (`trade_ticks`, no quotes): the tick rule, `side = sign(ΔP)`, flat
inherits the previous side. The research feature builder tags the column
`aggressor_rule = "tick"` so a model fitted on it is never confused with live
quote-rule features.

### 2.3 Rolling structures (fixed-length rings)

**Minute bins ring** — 60 bins of one minute each (a 61st slot is reserved so
the ring never needs resizing), indexed by `minute_index mod 61`:

```
bin = {t_open, buy_vol, sell_vol, n_trades, n_buy, n_sell,
       vwap_num (Σ price·size), vol (Σ size), high, low, close,
       large_vol (Σ size for prints ≥ large_print_floor)}
```

Per tick: locate the bin by minute, roll forward (zeroing skipped bins) if the
minute advanced, then add the trade: `buy_vol += size` or `sell_vol += size`,
`vwap_num += price·size`, update `high/low/close`, `large_vol += size` when
`size ≥ large_print_floor` (the symbol's 95th percentile print size over the
trailing session, refreshed once a minute). All O(1).

Window aggregates for 15 and 60 minutes are maintained as running totals,
updated by adding the current bin's increment and subtracting the bin that
fell out of the window on each minute roll (O(1), no rescans):

```
W15 = Σ_{bins in last 15 min} (buy_vol, sell_vol, vol, vwap_num, large_vol, n_trades)
W60 = same over 60 min
CVD_60 = Σ_{60 min} (buy_vol − sell_vol)            cumulative volume delta
CVD_15 = Σ_{15 min} (buy_vol − sell_vol)
```

**Quote ring** — the last 900 Level 1 updates (≈ 15 min at the observed
~1 update/s per name; capacity is a constant, not time-based, so memory is
bounded regardless of burst rate):

```
quote_ring = {ts, bid, ask, bid_size, ask_size}   capacity 900, head pointer
```

Two running EMAs are kept on top of the ring so no scan is needed:

```
imb_t   = (bid_size − ask_size) / (bid_size + ask_size)        ∈ [−1, 1]
IMB_ema = α·imb_t + (1−α)·IMB_ema,  α chosen so the half-life is 60 s at 1 Hz
spr_t   = (ask − bid) / mid
SPR_ema = β·spr_t + (1−β)·SPR_ema,  half-life 5 min
```

A `spread_baseline` per symbol (median `spr_t` over the trailing 20 sessions,
computed nightly from the quote ring's persisted minute medians) gives the
spread z-score. Quotes with `bid ≥ ask`, zero sizes or a crossed book are
dropped before touching the EMAs.

**Daily RSI distribution** — per symbol, the trailing 252 daily RSI(14) values
in a fixed array with an insertion-ordered copy for the percentile (one sort of
252 floats per *session*, not per tick). Values: `q05`, `q50`, `IQR`.

### 2.4 Persistence for research and the ledger

The minute bins are flushed to `data/micro/<symbol>/<date>.parquet` at the
close (one row per minute, ~390 rows/day). This is the forward dataset the
Level 1 component is fitted on; it is also what the shadow ledger joins to.

---

## 3. The turn score

Everything below is evaluated at the close of each 5-minute bar (the fit-lane
cadence), never per tick. Symbols are only scored while they are in the
oversold zone, so the work is bounded to a handful of names at a time.

### 3.1 Dynamic oversold zone (component Z)

```
r_t    = RSI14 of daily closes (lagged: yesterday's close at today's bars)
q05_t  = 5th percentile of {r_{t−252}, …, r_{t−1}}      (min 120 sessions, else q05 = 25)
IQR_t  = 75th − 25th percentile of the same window
zone_t = 1[r_t ≤ q05_t]
depth_t = clip((q05_t − r_t) / IQR_t, 0, 1.5) / 1.5      ∈ [0, 1]
Z_t = depth_t
```

No scoring happens unless `zone_t = 1`. MU, with a calmer RSI distribution
than SMCI, gets its own bar; in a bear tape the bar drifts lower on its own.

### 3.2 RSI momentum and divergence (component M)

RSI on 15-minute bars (`ρ_t`, 14 periods) for timing inside the daily zone:

```
slope_t      = ρ_t − ρ_{t−3}                             (three 15-min bars)
σ_slope      = std of slope over the trailing 20 sessions (floor 2.0)
inflect_t    = 1[slope_t > 0 and slope_{t−1} ≤ 0]
m_slope_t    = clip(slope_t / (2·σ_slope), −1, 1)

divergence over L = 20 fifteen-minute bars:
  P_low_now = min(low_{t−5..t}),   P_low_prev = min(low_{t−L..t−6})
  R_low_now = min(ρ_{t−5..t}),     R_low_prev = min(ρ_{t−L..t−6})
  div_t = 1[P_low_now < P_low_prev and R_low_now > R_low_prev]

M_t = clip(0.5·max(m_slope_t, 0) + 0.3·inflect_t + 0.2·div_t, 0, 1)
```

Negative slope contributes nothing (the component is one-sided by
construction: a falling RSI is not evidence of a turn).

### 3.3 Tape absorption, cumulative volume delta (component A)

```
sell_share_15 = sell_vol_15 / (buy_vol_15 + sell_vol_15)                 ∈ [0, 1]
ret_15        = (close_t − close_{t−15m}) / close_{t−15m}
ATR_15        = ATR14 on 15-minute bars, as a fraction of price
price_hold    = 1 − clip(−ret_15 / ATR_15, 0, 1)      1 when price did not fall, 0 when it fell a full ATR
absorb_t      = clip((sell_share_15 − 0.5) / 0.3, 0, 1) · price_hold
                sellers dominate the tape but price is not giving way → passive bids absorbing

cvd_turn_t    = 1[CVD_15 > 0 and CVD_60 < 0]            delta flipped positive inside a negative hour
cvd_slope_t   = clip(CVD_15 / (vol_15 + ε), −1, 1)       normalised 15-min delta
large_bias_t  = clip((large_buy_15 − large_sell_15) / (large_vol_15 + ε), −1, 1)

A_t = clip(0.5·absorb_t + 0.3·cvd_turn_t + 0.2·max(cvd_slope_t, 0), 0, 1)
      + 0.1·max(large_bias_t, 0)      (then clipped to [0, 1])
```

### 3.4 Level 1 book pressure (component B)

```
B_raw_t   = IMB_ema_t                                    ∈ [−1, 1]
spread_z  = (SPR_ema_t − spread_baseline) / spread_mad_baseline
liq_ok_t  = 1[spread_z ≤ 3 and bid_size + ask_size ≥ 2·median_bbo_size]
B_t       = liq_ok_t · clip((B_raw_t + 1) / 2, 0, 1)     maps [−1,1] → [0,1], 0 when the book is thin or wide
```

A wide or thin book is a *veto* of the whole score (§3.6), not merely a zero
component: the directive's "do not downgrade" applies to execution too.

### 3.5 Composite score

```
S_t = 100 · Σ_i w_i · c_i,   c = (Z, M, A, B),   w_i ≥ 0,   Σ w_i = 1
prior weights w0 = (0.30, 0.25, 0.25, 0.20)
```

`S_t ∈ [0, 100]` by construction because every component is in [0, 1] and the
weights are a convex combination. Each component is reported alongside the
score, so every decision reads as "zone 0.8, momentum 0.4, absorption 0.7,
book 0.3 → 61".

### 3.6 Decision rule

```
enter  iff zone_t = 1
        and S_t ≥ θ_regime            (θ per entry-time regime: risk_on / mixed / risk_off)
        and no veto: spread_z ≤ 3, bid_size + ask_size ≥ 2·median, not an earnings eve,
                     PriceBook mark fresh (≤ 10 s), BLOCK_NEW_ENTRIES off
entry  = next 5-min open (fill style per the family's entry_style)
stop   = entry − k·ATR14_daily,  k = 1.5 (research grid: 1.0 / 1.5 / 2.5)
target = max(20-day mean, entry + 1·risk)  (the mean must be ≥ 1R away, as today)
time   = 10 sessions (grid 5 / 10 / 15)
size   = base risk × (0.5 + 0.5·(S_t − θ)/(100 − θ))     scales from 0.5× at the threshold to 1× at 100
exit   = the existing PositionState FSM (breakeven → 1R scale → ATR trail)
         plus one new rule: thesis_failed when, within the first 2 sessions, S_t < θ − 20
         and CVD_60 < 0 (the turn did not materialise) → exit at market
```

### 3.7 Weight fitting — bounded, regularised, sign-constrained

Target per oversold occurrence (every 5-min bar with `zone_t = 1` is a
candidate row; rows within the same symbol-session are one *cluster*):

```
y = 1 if +1R (entry + risk) is touched before the stop within 10 sessions, else 0
```

Model: logistic regression on the standardised components with an L2 penalty
and non-negativity,

```
P(y=1 | c) = σ(b + Σ_i β_i · c_i),   β_i ≥ 0,   minimise  −loglik + λ‖β‖²
w_i = β_i / Σ_j β_j         (the composite's convex weights)
```

solved with bounded L-BFGS (`scipy.optimize.minimize`, bounds `[0, ∞)`), λ
chosen by grouped 5-fold cross-validation where the groups are symbol × month
(no symbol-month appears in both train and test). Final weights are shrunk to
the prior, `w = (1−η)·w_fit + η·w0`, with η the value that maximises
out-of-fold expected R; η is reported so "how much did the data move the
prior" is always visible. θ_regime is the score level that maximises
out-of-fold mean R subject to n ≥ 30 trades per regime; otherwise the regime
inherits the pooled θ.

Guardrails from the directive: the five-month archive is one regime, so the
archive fits only the components it can measure (Z, M, A); B is fitted only
from the forward ledger once it has ≥ 60 resolved occurrences; every refit is
monthly, writes `data/micro/turn_model.json` with weights, λ, η, fold scores
and the training window, and is applied only through the console's normal
loosen/tighten ratchet (a refit that *raises* exposure waits for the next
open).

### 3.8 Ledger row (shadow and live, one per oversold 5-min bar)

```
{ts, symbol, regime, r_daily, q05, depth, rho_15m, slope, inflect, div,
 sell_share_15, ret_15, absorb, cvd_15, cvd_60, cvd_turn, large_bias,
 imb_ema, spread_z, liq_ok, Z, M, A, B, score, theta, veto[], decision,
 entry_px, stop_px, target_px, outcome: {hit_1r, hit_stop, r_realised, sessions} (filled on resolution)}
```

Written to `data/shadow_fit_rsi_turn.ndjson` through the existing fit-lane
ledger path (extended with the factor block), resolved by the existing shadow
resolver.

---

## 4. Where it plugs in: a pluggable thesis rule

There is no `TradeThesis` object in the code today; the nearest seams are the
pure strategy *family* (`strategies/families/*`), the live `FitLaneStrategy`
adapter (`strategies/fit_lane.py`) that turns a family `Entry` into a
`TradingSignal`, and the `PositionState` FSM that owns exits. Phase 1 adds one
small protocol and keeps those seams:

```python
class TurnThesis(Protocol):
    name: str
    def evaluate(self, symbol: str, bars5: pd.DataFrame, ind: pd.DataFrame,
                 micro: MicroSnapshot, params: Mapping) -> ThesisVerdict: ...

@dataclass(frozen=True)
class ThesisVerdict:
    score: float                     # 0..100
    components: Mapping[str, float]  # Z, M, A, B and their raw inputs
    theta: float
    vetoes: tuple[str, ...]
    entry: Optional[Entry]           # None when not actionable
```

* **Research path.** A new family `rsi_turn` (`strategies/families/rsi_turn.py`)
  with the same `(bars5, ind, daily, news, params) -> list[Entry]` contract as
  every other family. The micro features arrive as columns on `ind`
  (`mf_cvd_15`, `mf_sell_share_15`, `mf_absorb`, `mf_imb_ema`, …) produced by a
  new builder `research/strategy_hunt/data/micro.py` from `trade_ticks` (tick
  rule) and, going forward, from the persisted minute bins (quote rule). The
  family calls the same `TurnScorer` as live, so research and live cannot drift.
* **Live path.** `FitLaneStrategy` gains an optional `micro_provider`; when
  the family is `rsi_turn` it attaches the `MicroSnapshot` columns to the live
  `ind` slice before calling the family. The verdict's components are copied
  into `TradingSignal.reasoning["thesis"]` and into the ledger row. Admission,
  playbook gate, desk limits, allocation cap and the console's block switch
  apply unchanged because the signal enters the router like any fit-lane signal.
* **Exits.** `PositionState` is untouched. `core/unified_exit.py` gets one
  thesis-aware rule, `thesis_failed`, consulted only for positions tagged
  `rsi_turn`, reading the latest `ThesisVerdict` from a per-symbol cache the
  scorer refreshes every 5 minutes.
* **Console.** `rsi_turn` is a lane like the others: shadow by default
  (`FIT_LANES_ENABLED`), live only via `FIT_LANE_LIVE_ENTRIES_ENABLED` and the
  allowlist, so enabling it is a next-open loosen.

---

## 5. Performance and concurrency

* Stream handler: one `on_tick` per Level 1 update, O(1): bin lookup by
  minute, running-total update, two EMA updates, ring head advance. No pandas,
  no allocation on the hot path (preallocated numpy arrays per symbol,
  ~61 bins × 12 fields + 900 × 5 quote slots ≈ 50 KB per symbol).
* Readers get `MicroSnapshot` copies produced under a short lock held by the
  stream task only during the array copy (microseconds).
* Scoring runs at 5-minute bar close in the fit-lane loop (already off the
  event loop via the bounded provider executor), only for symbols with
  `zone_t = 1`.
* The daily RSI percentile is recomputed once per session per symbol.
* The 15-minute RSI and ATR come from the same closed 5-minute bars the fit
  lane already loads; no extra provider calls.
* Ledger writes stay best-effort and append-only, as today.

---

## 6. Deliverables and order

1. `core/micro_book.py` (rings, aggressor tagging, snapshots, nightly flush) + tests (synthetic tick streams: quote rule cases, flat inheritance, bin roll, window totals vs brute force, bounded memory).
2. Stream wiring: pass sizes/volume from `core/schwab_stream.py` into the book; price book unchanged.
3. `core/turn_score.py` — pure functions for Z, M, A, B and the composite; property tests that every component is in [0, 1] and the score in [0, 100] for random inputs.
4. `research/strategy_hunt/data/micro.py` — tick-rule features from `trade_ticks` into 5-minute columns; `rsi_turn` family; pooled fixed-rule evaluation (as for `rsi_dip`) plus the weight fit with grouped CV; report under `research/reports/rsi/`.
5. `strategies/fit_lane.py` micro provider + ledger factor block; `unified_exit` thesis_failed rule; console registry entry; shadow enable after the close.
6. Docs: `docs/BOT_OPERATIONS_MANUAL.md` section, this spec linked from `docs/research/strategy_hunt.md`.

Not in Phase 1: Level 2, any change to sizing beyond the score scale, any live
enable.
