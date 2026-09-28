# Strategy hunt

Research and operations notes for the offline strategy-fit pipeline
(`research/strategy_hunt/`) and the engine-side gate that consults its
output. It covers the nightly scheduling of the offline hunt, the
flags that control the live engine gate, and — as of the Playbook Desk
B1 phase (2026-09-25) — the universe context builder, the thirteen
strategy families, the regime-conditioned and universe-pooled fit
rows, and the pluggable sentiment backend.

## Scheduling

The strategy hunt (`research.strategy_hunt`) runs offline, outside the
trading engine process, and writes its snapshot to
`data/strategy_hunt/fit/strategy_fit.json` (path configurable via
`STRATEGY_FIT_PATH`). It is scheduled to run nightly after the close,
via crontab, on this host:

```
30 22 * * 1-5 /home/nvidia/claude/grok/trading_bot_commentary/scripts/nightly_strategy_hunt.sh >> /home/nvidia/claude/grok/trading_bot_commentary/data/strategy_hunt/nightly/cron.log 2>&1
```

Runs 22:30 local, Monday–Friday, via `scripts/nightly_strategy_hunt.sh`
(added 2026-09-28 for the go-live restart; supersedes an earlier crontab
line that invoked `research.strategy_hunt all` directly at 22:00). The
script `cd`s into the repo, sources `.env` for API credentials and
`POSTGRES_DSN` (falling back to `backtest.engine.DEFAULT_DSN` if unset),
then runs `python -m research.strategy_hunt all --workers 12 --postgres`,
appending its own start/end banner lines plus the pipeline's stdout/stderr
to `data/strategy_hunt/nightly/<date>.log`; the crontab redirect above
(`nightly/cron.log`) only catches anything that escapes that per-date log
(e.g. a failure before the script's own logging starts). Installing the
crontab line is an operator step (`crontab -e`); it is not applied
automatically by any code change. A test run on 2026-09-28 completed in 17
minutes (`rc 0`).

**Since Playbook Desk B2 (2026-09-25), `all` ends with `playbook`** (see
"Run" below): the nightly script's `--postgres` run therefore rewrites
`data/desk/playbook.json` every night from the shadow ledgers **and** the
live `bot_trades` tape — with no `--postgres` the nightly playbook rebuild
would run from the shadow ledgers alone. See "The `playbook` subcommand"
below.

`GET /api/strategy-fit` (read-only) and the "Strategy fit" section of
`tools/bot_monitor.py`'s report both read the snapshot this job
writes, so a missed or failed cron run shows up as a stale or
`_unavailable_` reading in the morning monitor check rather than
silently going unnoticed.

## Engine gate

The strategy-fit gate (`core/strategy_fit_gate.py`, wired into
`core/engine.py` right after the regime gate) checks each governed
signal's (symbol, strategy-family, side) verdict against the
strategy-fit snapshot before letting it proceed. It is off by default
everywhere it is currently wired — see
`docs/BOT_OPERATIONS_MANUAL.md` § "Strategy fit gate" for the
pipeline position, audit line format, and fail-closed behavior.

**Snapshot caveat (2026-09-25, B1.5 run).** The current full-universe snapshot contains **0 PASS rows** — none
of its 9,231 unconditional rows (539 symbols × 13 families), none of its 1,390 symbol-regime rows, none of its
69 universe-pooled regime rows (see "Timing" below and
`research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md`). Turning `STRATEGY_FIT_REGIME_ROWS` on or off
therefore changes nothing today: there is no regime row that would let a lane through. So with today's
snapshot, setting
`STRATEGY_FIT_SHADOW=False` while `ENABLE_STRATEGY_FIT_GATE=True` makes **every governed lane skip
every signal** — `fit_fail` for the symbols that have a row, `fit_missing_row` for the rest. That is
not a malfunction: enforcing mode is a deliberate *"governed lanes off until the evidence exists"*
decision, and the governed lanes are `mean_reversion`, `mean_reversion_short`,
`oversold_bounce_v2`, `day_trade_momentum`, `day_trade_momentum_short`, `orb_contraction_rvol`,
`news_sentiment`, `free_news_sentiment`, plus any `fit:<family>` lane. Turn shadow off only when you
intend exactly that, or once a nightly run has produced PASS rows for the symbols you trade.

| Flag | Default | Purpose |
|---|---|---|
| `ENABLE_STRATEGY_FIT_GATE` | `False` | Master switch. Gate is fully inert (no audit lines, no blocking) until this is `True`. |
| `STRATEGY_FIT_SHADOW` | `True` | When the gate is enabled, run it in shadow (log the decision, never block a live entry). Must be explicitly set `False` to let the gate skip signals. |
| `STRATEGY_FIT_PATH` | `data/strategy_hunt/fit/strategy_fit.json` | Path to the snapshot file the gate and `/api/strategy-fit` read. |
| `STRATEGY_FIT_MAX_AGE_DAYS` | `7` | Snapshot age (days) beyond which it is treated as stale. |
| `STRATEGY_FIT_REGIME_ROWS` | `True` | Let the gate consult the regime-conditioned rows (and the universe-pooled regime lanes) for the live market-context regime instead of the unconditional row. See "Regime and pooled rows" below. Set `0`/`false` to restore the pre-B1 unconditional-only lookup. |
| `SENTIMENT_BACKEND` | `vader` | Sentiment scorer used by both the research news cache and live news scoring (`core/sentiment_backend.py`), so the two agree. `vader` or `finbert`. |
| `SENTIMENT_FINBERT_PATH` | `models/finbert` | Directory holding the locally prepared FinBERT checkpoint (`python scripts/prepare_finbert.py`). Falls back to the hub id when incomplete, and to VADER when FinBERT cannot load at all. |
| `FIT_LANES_ENABLED` | `""` (empty) | Comma-separated family names to register live strategy-fit lane adapters for. No lanes registered by default. |
| `FIT_LANE_LIVE_ENTRIES_ENABLED` | `""` (empty) | Comma-separated family names allowed to place LIVE entries via their fit lane; a family absent here is shadow-only even if present in `FIT_LANES_ENABLED`. |
| `SWING_ALLOW_OVERNIGHT` | `False` | Allow swing-lane positions to hold across sessions instead of forced flat by EOD. |
| `SWING_MAX_HOLD_SESSIONS` | `10` | Maximum sessions a swing-lane position may be held before a forced exit. |
| `STRATEGY_HUNT_DATA_ROOT` | `data/strategy_hunt` | Root of the nightly research data tree. The engine splices `<root>/bars5/<SYM>.parquet` and `<root>/daily/<SYM>.parquet` under the broker's short recent window so `rvol` (20 prior sessions) and `or*_range_avg20` (10) can resolve live; also the root for `<root>/earnings/<SYM>.parquet` in the swing exit checks. |

All twelve flags follow the config manager's env-var-then-yaml-then-default
precedence (`env ENABLE_STRATEGY_FIT_GATE=1`, yaml
`trading.enable_strategy_fit_gate: true`, etc. — see
`core/config.py`, search `v-strategy-fit-2026-09-25` for the first nine and
`v-playbook-desk-b1-2026-09-25` for the last three).

When `STRATEGY_FIT_REGIME_ROWS` is on, `core/engine.py` passes the live
`market_context_regime` from the signal's reasoning into
`core/strategy_fit_gate.py`, which uses `StrategyFit.lookup_regime` instead of
`lookup` for the three real regime labels (`risk_on`, `risk_off`, `mixed`) and
adds `regime=` and `pooled=` to the audit line. Any other value (including a
missing regime) means "no regime evidence to use" and the unconditional row
governs, exactly as before B1.

## Run

The pipeline is a sequence of subcommands under `python -m research.strategy_hunt`, each
reading/writing under `--root` (default `data/strategy_hunt/`):

    set -a; . ./.env; set +a
    PY=/home/nvidia/anaconda3/envs/trading-bot/bin/python
    $PY -m research.strategy_hunt export --min-days 250        # Postgres minute_bars -> bars5/daily Parquet
    $PY -m research.strategy_hunt news --symbols NVDA,TSLA      # Alpaca news, incremental
    $PY -m research.strategy_hunt news --rescore                # re-score the cached news with SENTIMENT_BACKEND
    $PY -m research.strategy_hunt earnings --symbols NVDA,TSLA  # yfinance earnings dates + EPS surprises
    $PY -m research.strategy_hunt context --symbols NVDA,TSLA   # SPY/VIX regime inputs + splits + universe context
    $PY -m research.strategy_hunt universe-context              # rebuild only context/universe_daily.parquet
    $PY -m research.strategy_hunt calibrate                     # realized fill slippage from Postgres fills
    $PY -m research.strategy_hunt run --workers 12              # walk-forward per symbol; incremental, safe to rerun
    $PY -m research.strategy_hunt run --force                   # recompute everything
    $PY -m research.strategy_hunt run --shuffle-seed 7          # shuffled baseline, in-root (trades_shuffle/)
    $PY -m research.strategy_hunt score --no-coldstart          # verdicts + fit snapshot + report
    $PY -m research.strategy_hunt report                        # rebuild the report from an existing fit snapshot
    $PY -m research.strategy_hunt scorecard --family orb        # Stage A shadow-resolver scorecard for one family
    $PY -m research.strategy_hunt promote                       # promotion table + .env lines for every family
    $PY -m research.strategy_hunt all --workers 12              # export -> news -> earnings -> context -> calibrate -> run -> run(shuffle) -> score
    $PY -m research.strategy_hunt all --workers 12 --no-baseline   # ...without the shuffled baseline run

Since v-playbook-desk-b1-5 the **shuffled baseline lives inside the main root**: `run --shuffle-seed N`
writes `trades_shuffle/`, `checkpoints_shuffle/` and `choices_shuffle/` next to the real tape's
directories and shares the root's bars, news, earnings and context, so no second root and no
symlinking are needed. `score` picks it up automatically whenever `trades_shuffle/` exists and pools
it into baseline rows the real pooled rows must beat (see "Regime and pooled rows"). `all` runs it
for you with seed 7 right after the real `run` — and **`all` rejects `--shuffle-seed`** with a
parser error (B1.6), since a user-supplied seed would turn its real leg into a second shuffled run
and `trades/` would never be written; `--no-baseline` skips it, in which case whatever
baseline is already in the root is reused — `score` checks it rather than trusting it, and flags
`baseline_mismatch` (the control covers less than 95% of the real tape's symbols) or
`baseline_stale` (its last exit is more than 7 calendar days behind the real tape's) on every
pooled row, logging the counts and dates. Only a root with no `trades_shuffle/` at all yields
`no_baseline`.

This in-root baseline **supersedes the old separate-root shuffle control**: there is no longer a
`fit/strategy_fit_shuffle.json`, and the report's whole-run false-discovery ratio is replaced by the
per-lane `excess` column in "Pooled rows vs shuffled baseline", which is the sharper question.

Controls for cost sensitivity use `--cost-mult 0.5` / `2.0` with their own `--root`, then copy the
JSON to `fit/strategy_fit_cost_0.5.json` / `fit/strategy_fit_cost_2.0.json` before `score` so the
report picks them up. A control root needs the same inputs as the main root — symlink or copy
`bars5`, `daily`, `news`, `earnings`, `splits` **and `context`** into it (the last one is required
since B1: without `context/universe_daily.parquet` the three in-play families see all-NaN `ctx_*`
columns and propose nothing).

`all` is what the nightly cron job (see "Scheduling" above) runs, in the order
`export → news → earnings → context → calibrate → run → run(shuffle) → score → playbook`: `context`
must run *after* `news`/`earnings` so the universe context's `ctx_news_count_24h`,
`ctx_news_sent_24h`, `ctx_earnings_prev` and `ctx_eps_surprise_pct` reflect tonight's caches by the
time `run` reads them, the shuffled baseline run must land before `score` reads both tapes
(`--no-baseline` skips it), and `playbook` runs **last** (Playbook Desk B2, 2026-09-25) because it
reads the fit verdicts `score` just wrote (`core.desk.playbook.update_entry`'s `fit_verdict` input,
via `best_verdict` in `research/strategy_hunt/playbook_update.py`). `all`'s own `--postgres` flag
passes straight through to `playbook` (`cmd_all` calls `cmd_playbook(a)`), so `all --postgres` also
folds the live `bot_trades` tape into that night's playbook rebuild. The other subcommands are for
ad hoc reruns of one stage (e.g. re-running `score` after editing the verdict floors without
re-simulating, or `universe-context` after a news backfill).

### The `playbook` subcommand

`python -m research.strategy_hunt playbook --root <root> [--families f1,f2,...]
[--desk-dir data/desk] [--postgres]` (Playbook Desk B2, 2026-09-25) rewrites
`<desk-dir>/playbook.json` (default `data/desk/playbook.json`, `core.desk.playbook.DEFAULT_DESK_DIR`)
from every resolved trade it can find:

- **Shadow tape** — always read: each family's shadow ledger
  (`data/shadow_fit_<family>.ndjson`, `research/strategy_hunt/shadow_resolver.py::resolve`),
  labelled with the entry-time regime (the same `RegimeLookup` the B1.6 fix introduced) because the
  resolver itself leaves every trade `unknown`.
- **Live tape** — only with `--postgres` (needs `POSTGRES_DSN` in the environment): `bot_trades`
  rows with a non-null `pnl_r`, mapped to a family through a research-side copy of the live map
  (`research/strategy_hunt/playbook_live.py::FAMILY_BY_STRATEGY`, pinned equal to
  `core.strategy_fit_gate.LANE_MAP` by `tests/research/strategy_hunt/test_playbook_update.py`),
  regime joined from the nearest preceding `bot_decisions` audit row (`strategy_fit`/`playbook`
  components) within `REGIME_MATCH_WINDOW` (30 minutes), else `unknown`. Without `--postgres` the
  update runs from the shadow ledgers alone — `all` (no flag) still rewrites the file every night,
  just without the live trades folded in.

Per `(family, side, regime)` key it takes the last `RECENT_WINDOW=60` resolved trades within the
last `RECENT_DAYS=90`, applies `core.desk.playbook.update_entry` against that key's best fit
verdict in tonight's snapshot (`best_verdict`: pooled row, then per-symbol regime rows, PASS >
WATCH > FAIL), and runs the result through `carry_forward` so a `suspended` key with no resolved
trade in `SUSPENSION_LAPSE_DAYS=30` days lapses back to `probation`. A key with no trades in the
window simply keeps its previous entry (still subject to the lapse check). See `docs/desk.md` §
"Playbook rules" for the transition table and `docs/BOT_OPERATIONS_MANUAL.md` § "Desk (shadow)" for
how the live engine reads the file it writes.

Console output is one line per key (worst status first) plus a summary, e.g.:

```
mean_rev_intraday   long  risk_on  probation x0.50 n=  7 exp_r=-0.080 t=-0.60 probation_small_sample
playbook: 1 key(s) {'probation': 1} -> data/desk/playbook.json
```

## Outputs

| Path | Content |
|---|---|
| `data/strategy_hunt/bars5/<SYM>.parquet`, `daily/`, `news/`, `earnings/`, `splits/` | Inputs (git-ignored). `news/<SYM>.parquet` carries `sentiment_vader`, `sentiment_finbert` and the effective `sentiment`; `earnings/<SYM>.parquet` carries `eps_estimate`, `reported_eps`, `surprise_pct` |
| `data/strategy_hunt/context/universe_daily.parquet` | Universe context: one row per (symbol, date) with the `ctx_*` feature set and its cross-sectional ranks (see "Universe context") |
| `data/strategy_hunt/trades/<SYM>.parquet` | All out-of-sample trades for the symbol |
| `data/strategy_hunt/choices/<SYM>.json` | Chosen parameters per window and family |
| `data/strategy_hunt/checkpoints/<SYM>.json` | Skip marker for incremental runs |
| `data/strategy_hunt/trades_shuffle/`, `choices_shuffle/`, `checkpoints_shuffle/` | Same three, for the shuffled baseline variant (`run --shuffle-seed N`). `score` pools `trades_shuffle/` into the baseline rows pooled verdicts are measured against |
| `data/strategy_hunt/calibration.json` | Realized slippage summary (`calibrate`); used as a fixed slippage override when usable |
| `data/strategy_hunt/fit/strategy_fit.json` | Latest fit snapshot (what the engine gate and `/api/strategy-fit` read); since B1 it also carries the `regime_rows` and `pooled_rows` sections, and since B1.5 `baseline_pooled_rows` (the shuffled control's pooled rows, which the live reader ignores). Non-finite floats (`recovery` +inf, `excess_r` NaN) are written as `null` so the file is valid JSON |
| `data/strategy_hunt/fit/fit_<date>.parquet`, `fit_regime_<date>.parquet`, `fit_pooled_<date>.parquet`, `fit_baseline_<date>.parquet` | Same rows as Parquet, one file per row family. Only `fit_<date>.parquet` goes to the Postgres `strategy_fit` table — its key is (symbol, strategy, side, computed_at), which pooled rows would collide on across regimes |
| `research/reports/strategy_hunt/<date>T<HHMM>-strategy-hunt.md` | Report (verdict table, pooled-vs-baseline table, pooled metrics, portfolio replay, cost calibration). Stamped with `computed_at` since B1.5, so two runs in one day no longer overwrite each other |
| `research/reports/strategy_hunt/<date>T<HHMM>-strategy-hunt.json` | Report manifest (row/symbol counts, verdict counts, pooled verdict counts), same stem as the report |
| `research/reports/strategy_hunt/<date>.json` | Run manifest (symbol/family counts, elapsed, errors), written by `run`; the shuffled baseline run records itself under `run_shuffle` / `errors_shuffle` so it cannot overwrite the real run's entry or its error list |
| `research/reports/strategy_hunt/<date>T<HHMM>-equity-curve.csv` | Monthly equity curve from the portfolio replay |
| `research/reports/strategy_hunt/scorecard-<family>-<date>.md` | Stage A shadow-resolver scorecard for one family |
| `research/reports/strategy_hunt/strategy_hunt_promotion.md` | Promotion table (all families) + the `.env` lines to apply, rewritten each `promote` run. Written under `reports_dir(root)` (B1.6), so a control root writes `<root>/reports/` instead of overwriting the committed one |
| `data/desk/playbook.json` (default; `--desk-dir` overrides) | Playbook Desk B2 (2026-09-25): the adaptive playbook, rewritten by `playbook` (and by `all`, which now ends with it). Read by the live engine's `playbook` router hook and `GET /api/desk/playbook`; see `docs/desk.md` § "Playbook rules" |

## Families

Thirteen families (`strategies/families/registry.py`, `family_names()`), each simulated long and
short where the family defines both sides:

| Family | Module | Horizon | Sides |
|---|---|---|---|
| `dt_momentum`, `orb`, `vwap_reversion`, `gap_play` | `intraday_momentum.py` | intraday | long, short |
| `mean_rev_intraday` | `intraday_reversion.py` | intraday | long |
| `news_catalyst`, `post_news_drift` | `news.py` | intraday | long, short |
| `pullback_trend`, `mr_20d`, `vol_breakout` | `swing.py` | multiday | `pullback_trend` long; others long, short |
| `orb_inplay` | `inplay.py` | intraday | long, short |
| `pead_smallcap` | `inplay.py` | multiday | long, short |
| `rs_leader` | `inplay.py` | multiday | long |

The last three are the B1 additions, and they are the only families that need the universe context:

- **`orb_inplay`** — opening-range breakout restricted to the day's in-play names. Evaluated once at
  `bar_of_day == 1` (the 09:35 bar, i.e. after the first bar has closed) on symbols whose
  `ctx_rvol_rank5` is in the top `top_n` (10/20/30) of the same session's cross-section, with a
  bar-0 close ≥ $5 and `ctx_atr14_pct` ≥ 0.5. Long when bar 0 closed up and price takes out bar 0's
  high, short on the mirror; stop is `stop_atr` ATRs beyond the entry but never further than bar 0's
  opposite extreme, and the position is time-stopped at the 15:55 bar. **The simulator has no
  trailing stop**, so the `trail_r` parameter is approximated as a take-profit at `trail_r` R
  (`trail_r = 0` means "no target": a 20 R placeholder). A trailing-stop simulator option is a
  follow-up, and the family's measured numbers should be read as a lower bound on a real trail.
- **`pead_smallcap`** — post-earnings-announcement drift for small/mid caps. Fires at
  `bar_of_day == 6` (the 10:00 bar has closed) on a session following an earnings date
  (`ctx_earnings_prev`), with `ctx_adv_bucket` ≤ mid, `|ctx_gap_pct|` ≥ `min_gap`, an EPS surprise of
  at least ±`s` percent in the trade's direction, and price on the right side of the session open.
  1.5-ATR stop, 3 R target, `hold_days` (5/10/20) session time stop, one entry per `hold_days`.
- **`rs_leader`** — relative-strength leader pullback-resume. At a session open, requires
  `ctx_rs_rank_60` ≥ `rs_floor` (0.85/0.90/0.95), `ctx_pct_from_52w_high` ≥ −15%, the prior daily
  bar straddling its EMA(10/20) (`d_low ≤ ema ≤ d_close`) and a positive prior-day return.
  `atr_stop` daily ATRs of risk, 3 R target, 20-session time stop.

All three read `ctx_*` through `strategies/families/context.py`, whose accessor returns all-NaN when
a column is absent — so without a context file they simply propose nothing rather than failing.

## Universe context

`research/strategy_hunt/context.py` builds `context/universe_daily.parquet`: one row per (symbol,
session) with the `CTX_COLUMNS` feature set, joined onto each symbol's indicator frame as `ctx_*`
columns by `indicator_frame(context=...)`. `context` builds it as its last step (after splits);
`universe-context` rebuilds only this file, taking just `--root`.

| Column | Meaning |
|---|---|
| `rvol_open5`, `rvol_open15` | Session's first-bar / first-three-bar volume over the mean of the prior 14 sessions' same measure (NaN under 5 prior sessions) |
| `gap_pct` | Open vs. prior close, percent |
| `rs_20`, `rs_60`, `rs_120` | Symbol return minus SPY's over the prior n sessions, percent |
| `rs_rank_60` | Same-date cross-sectional rank of `rs_60` |
| `rvol_rank5` | Same-date cross-sectional rank of `rvol_open5` (1 = most in-play) |
| `pct_from_52w_high` | Distance from the 252-session high, percent (needs ≥ 126 sessions) |
| `adv_dollars_20`, `adv_bucket` | 20-session average dollar volume and its bucket (`mega` ≥ $1B, `large` ≥ $200M, `mid` ≥ $20M, else `small`) |
| `atr14_pct` | ATR(14) as a percent of price |
| `earnings_today`, `earnings_prev` | An earnings date on this session / in the prior session's window |
| `eps_surprise_pct` | Surprise of the most recent earnings report in that window (from the earnings cache) |
| `news_count_24h`, `news_sent_24h` | Count and mean sentiment of headlines in the 24 hours before the session open |

**Known-at-open semantics.** Everything except `rvol_open5`, `rvol_open15` and `gap_pct` is computed
through the *prior* close, so it is genuinely known before the session opens. Those three use the
current session's opening bars and are therefore only consumed from `bar_of_day >= 1` onwards —
`orb_inplay` reads `ctx_rvol_rank5` at bar 1, after the bar-0 volume that feeds it has closed.
`rvol_rank5` inherits the same restriction because it ranks `rvol_open5`; `rs_rank_60` and
`adv_bucket` rank already-lagged inputs and add no look-ahead of their own. Every input is loaded
through `data.splits.load_adjusted`, so the context is split-adjusted like the rest of the pipeline.

A missing or empty SPY frame aborts the whole build (every symbol's `rs_*` would silently be
all-NaN otherwise); a single other symbol's failure is logged and skipped. `cmd_context` treats a
failed context build as fatal rather than continuing into `run` with a stale file.

**Live freshness requirement (go-live, 2026-09-28).** The live fit lane for
context-consuming families (`mom_12_1`, `low_vol_trend`, the in-play
families) loads this file at signal time via
`core/strategy_fit_bars.py::load_context_slice` and maps each session onto
its own row via `strategies/fit_lane_frames.py::align_context`. A session
with no row of its own (e.g. today, before the next nightly build runs)
carries forward the most recent earlier row for at most
`CTX_MAX_CARRY_SESSIONS = 3` business days — beyond that the whole row goes
NaN and the family stops proposing entries (fails safe rather than trading a
stale cross-section). On a carried row, only the slow prior-close factors
carry (`mom_12_1`, `mom_rank_12_1`, `vol_rank_60`, `rs_*`, `adv_*`,
`atr14_pct`, `pct_from_52w_high`); the session-local columns
(`rvol_open5`, `rvol_open15`, `rvol_rank5`, `gap_pct`, `earnings_today`,
`earnings_prev`, `eps_surprise_pct`, `news_count_24h`, `news_sent_24h`) are
NaN'd because they describe a session that has not happened yet. In
practice: **if `fit:mom_12_1` stops signalling entirely, check
`data/strategy_hunt/nightly/<date>.log` and the mtime of
`context/universe_daily.parquet` first** — a nightly-cron outage longer than
3 sessions (a long weekend plus a missed Monday run, for instance) silences
the family until a fresh build lands.

## Regime and pooled rows

`research/strategy_hunt/score/regime_rows.py` adds two row families to the snapshot on top of the
unconditional per-(symbol, family, side) rows, both written by `score` and read by
`core/strategy_fit.py::lookup_regime`:

- **Regime rows** — one row per (symbol, family, side, regime) for each of `risk_on`, `risk_off`,
  `mixed`, scored against the same floors as the unconditional row. A symbol that only works in
  `risk_on` shows PASS there and FAIL elsewhere, which is what lets the gate stand a lane down when
  the tape turns. A regime with fewer than `REGIME_ROW_MIN_N` (= 60, the WATCH floor) trades gets
  **no row at all**: below that, `stability`/`decay` are structurally 0 and the row could only ever
  come out FAIL, and that manufactured FAIL would outrank both the pooled and the unconditional
  evidence. No evidence is not a FAIL. Those trades still count in the pooled and unconditional rows.
- **Pooled rows** — one row per (family, side, regime) over the whole universe, with `symbol = "*"`
  and `source = "pooled"`, scored against the stricter `POOLED_FLOORS`: n ≥ 500, sessions ≥ 60,
  PF ≥ 1.2, WR ≥ 45%, exp_R ≥ +0.05, recovery ≥ 3.0, stability ≥ 0.6, decay ≥ 0.5, t_stat ≥ 3.0,
  t_stat_session ≥ 3.0, exp_r_lb ≥ 0 and **excess over the shuffled baseline ≥ +0.05 R**. A pooled
  PASS is the family-level claim "this lane works in this regime", which lets a lane trade a symbol
  with no history of its own. Pooled `window_id` is re-bucketed to the **calendar quarter** of the
  entry before metrics are computed, since a per-symbol walk-forward window index is not
  contemporaneous across symbols — so a pooled PASS needs at least four calendar quarters with
  trades for `stability`/`decay` to clear their floors. There is no `max_dd_r` cap: a flat R
  drawdown ceiling is not scale-invariant across pooled trade counts (B1 saw a 13k-trade row with a
  639 R drawdown by arithmetic alone), so `min_recovery` (net R / max DD R) asks the scale-invariant
  question instead.
- **Baseline rows** (v-playbook-desk-b1-5) — the same pooled rows computed over the *shuffled*
  tape in `<root>/trades_shuffle/`, stamped `source = "baseline"` and stored under the snapshot's
  own `baseline_pooled_rows` key. Every real pooled row carries `baseline_exp_r`, `baseline_n` and
  `excess_r = exp_r − baseline_exp_r`, and the `min_excess_r` floor is what stops a pooled row from
  passing on the regime's drift: the B1 run's shuffle control reached a pooled PASS while the real
  tape did not. With no matching baseline group the excess is NaN, the row records the
  `no_baseline` failure, and it can be WATCH but **never** PASS. A baseline that is present but
  covers less than 95% of the real tape's symbols, or whose last exit is more than 7 calendar days
  behind it, is flagged `baseline_mismatch` / `baseline_stale` on every pooled row
  (`score/regime_rows.py::baseline_guard`) — a leftover control silently becoming the yardstick is
  worse than having none. A baseline *group* whose trade count is below 0.5x or above 2x the real
  row's `n` is flagged `baseline_population_mismatch` (B1.6): the two rows are then measuring
  different populations, so their expectancy difference is not an excess over drift. That key also
  forbids PASS, `baseline_n` is rendered next to `baseline exp_R` in the report's pooled table, and
  the report's JSON manifest counts the flagged rows as `pooled_baseline_population_mismatch`.
  Baseline rows never appear in `pooled_rows`/`regime_rows` and the live gate never reads them.

### How a trade's regime label is computed (entry-time only)

`research/strategy_hunt/data/regime.py` builds one `RegimeLookup` per symbol run from the root's
**5-minute SPY bars** and **daily VIX** closes, and the simulator stamps every trade with
`lookup(entry_ts)` where `entry_ts` is the fill bar's timestamp. The three inputs are the same ones
`core/market_context.py` uses live, through the same `classify_regime` thresholds, but reconstructed
from data that existed *at the entry bar*:

| input | value |
|---|---|
| `spy_change_pct` | SPY's day-to-date move: the close of the last 5-minute SPY bar that closed at or before the entry bar's open, against that session's open. `0.0` for an entry on the session's first bar (nothing has closed yet). |
| `vix_level` | the **prior** session's VIX close. |
| `vix_change_pct` | the **prior** session's close-to-close VIX change. |

Until B1.6 the label was computed from the entry *day's* SPY and VIX **closes**, which is a
look-ahead: a 10:00 entry was stamped `risk_off` because SPY closed −1% six hours later, and every
regime row (and the gate reading it) was fitted to information the desk could not have had. The
shuffle control shares the same lookup object, since SPY is never shuffled — only the symbol's own
tape is — so the control is measured against the same labels as the real tape.

The VIX pair is a documented **lagged proxy**: the research store holds daily VIX only, so an
intraday level is unavailable. Live, `vix_change_pct` is today's move; here it is yesterday's.
Lagging can never smuggle in the future, but it does make a regime row a blunter instrument than
the live label on days when VIX turns intraday.

Trades whose regime is `unknown` (no SPY bars for that session, or no prior VIX session to take a
close-to-close change from) are dropped from both row families — they would describe a data gap,
not a regime — and still count in the unconditional row.

**Lookup precedence** (`StrategyFit.lookup_regime`, used by the gate when
`STRATEGY_FIT_REGIME_ROWS` is on and the live regime is one of the three labels):

1. the symbol's own row for that regime,
2. the universe-pooled row for that regime — **only when the symbol's own unconditional row for
   that lane is absent or is not FAIL** (B1.6),
3. the unconditional row for the lane,
4. `None` (which the gate treats as `fit_missing_row`).

Step 2's restriction is the difference between filling a gap and overruling evidence. A pooled PASS
says "this lane works in this regime somewhere in the universe"; it may stand in for a symbol the
pipeline has never measured, and it may stand in front of a WATCH ("not measured well enough yet"),
but it must never outrank the pipeline's own measured FAIL for that symbol and lane.

A stale, missing or unloaded snapshot yields `None`, exactly as `lookup` does.

**Reading `t_stat` honestly.** Pooled `t_stat` is a trade-level mean-R t-statistic that treats every
trade as independent. Pooled trades include many same-session entries across correlated symbols, so
the effective sample is smaller than n suggests and `min_t_stat 3.0` is a weaker filter than it
reads — which is why `min_t_stat_session` now clusters by session (the true independent unit) on top
of it, and `min_excess_r` requires the row to beat its own shuffled control. Treat a pooled PASS as a
hypothesis worth shadow-trading, not a measured edge.

## Sentiment backend

`core/sentiment_backend.py` is the single scorer for both the research news cache and live news
scoring, so research and live can never disagree about a headline:

- `SENTIMENT_BACKEND=vader` (default) — CPU, always available.
- `SENTIMENT_BACKEND=finbert` — ProsusAI/finbert on GPU when available. The hub copy ships only a
  pickle checkpoint, which transformers ≥ 5 refuses to load on torch < 2.6, so run
  `python scripts/prepare_finbert.py` once to write a safetensors checkpoint to
  `SENTIMENT_FINBERT_PATH` (`models/finbert`). If transformers/torch/the model are missing the
  backend falls back to VADER with a one-time warning rather than failing.

The news cache keeps both scores side by side: `sentiment_vader` always, `sentiment_finbert` when
the configured backend is finbert and it actually loaded, plus an effective `sentiment` column
(finbert where present, VADER per-row where not) — that effective column is what
`load_news`, the universe context's `news_sent_24h`, and the `news_catalyst`/`post_news_drift`
families read, so nothing downstream re-scores anything.

    SENTIMENT_BACKEND=finbert $PY -m research.strategy_hunt news             # fetch + score new rows
    SENTIMENT_BACKEND=finbert $PY -m research.strategy_hunt news --rescore   # backfill the existing cache

`news --rescore` needs no Alpaca credentials: it fills `sentiment_vader` for any unscored row, adds
and fills `sentiment_finbert` when finbert is the effective backend, recomputes `sentiment`, and
writes only when something actually changed. It scores in chunks of 2,048 headlines so a large
symbol file does not build one giant result list. Switching the backend and re-scoring changes the
inputs to two families and one context column, so it invalidates previous results for them —
re-run `run --force` afterwards.

## Verdicts

PASS requires n ≥ 150, sessions ≥ 10, PF ≥ 1.30, WR ≥ 48%, exp_R ≥ +0.05, max DD ≤ 20 R,
stability ≥ 0.6, decay ≥ 0.5, t_stat ≥ 2.0, and exp_r_lb ≥ 0.0 (see
`research/strategy_hunt/score/verdict.py`, `FLOORS`). The last two floors are statistical-significance
guards on top of v1's original eight: `t_stat` is the trade-level mean-R t-statistic and `exp_r_lb` is
the lower bound of its confidence interval, so a (symbol, family, side) combo cannot PASS on a lucky
small sample with a wide confidence interval even if its point-estimate exp_R clears +0.05.

WATCH requires n ≥ 60, exp_R > 0, and PF ≥ 1.1 but falls short of at least one PASS floor. Cold-start
rows (symbols below `COLDSTART_MAX_N` walk-forward trades, scored by the trained classifier in
`research/strategy_hunt/fit/coldstart.py` instead of the floors above) can only be WATCH or FAIL —
never PASS — since there isn't enough direct evidence yet. Everything else is FAIL.

## Split handling

`research/strategy_hunt/data/splits.py` fetches each symbol's split history (default fetcher:
yfinance) via `update_splits` and caches it to `data/strategy_hunt/splits/<SYM>.parquet` as
`(date, prior_multiplier)` rows — `prior_multiplier` back-adjusts all bars *before* the split date so
price levels are continuous across it. `detect_splits` independently flags likely splits straight from
the daily bars (a large overnight open/close ratio against one of the common split ratios, with
same-day intraday move capped so ordinary gaps aren't mistaken for splits) as a sanity cross-check
against the fetched history. `run.py`'s `process_symbol` always loads bars through
`data.splits.load_adjusted`, so every family sees split-adjusted prices; entries, stops, and R-multiples
are computed on adjusted prices throughout.

## Costs

`research/strategy_hunt/sim/costs.py`'s `CostModel` prices every fill with:

- **Spread** — a monthly half-spread estimated from the Corwin-Schultz high/low estimator
  (`corwin_schultz_spread`), clipped to `[0.0002, 0.005]` and looked up by the entry/exit month.
- **Slippage** — volume-scaled by default: `0.0001 * participation_pct` where `participation_pct` is
  the requested notional's share of the bar's dollar volume, floored at `0.0001`. `calibrate` can
  replace this with a fixed `slippage_override` fraction derived from realized (broker fill vs.
  decision price) slippage in Postgres, when the sample is large/consistent enough
  (`calibration_summary`'s `usable` flag; `slippage_override_from` reads it back for `run`).
- **Sell fees** — a flat `2.946e-5` of notional on every sell (SEC/TAF-style), long exits and short
  entries alike.
- **Borrow** — 3% annualized (`BORROW_RATE_ANNUAL = 0.03`), charged only on short positions,
  prorated by `hold_sessions / 252` and added to the short's exit fill.

`--cost-mult` scales spread, slippage floor, borrow rate, and any slippage override together
(`CostModel.scaled`) for the cost-sensitivity control runs described under "Run" above.

## Portfolio replay

After scoring, `report.py` filters all pooled trades down to the (symbol, family, side) combos with a
walk-forward PASS verdict (`portfolio.pass_lane_trades`) and replays them chronologically through the
bot's real position-sizing rules (`portfolio.SizingRules`: $43,500 starting equity, 1.5% risk per trade,
0.5 Kelly fraction, day-trade risk halved, $10,000 / 20%-of-buying-power notional cap, 5 concurrent
positions, 0.69 buying-power ratio) to produce a dollar equity curve — a sanity check that the PASS
lane, taken together under real capacity constraints, is still profitable and not just a sum of
individually-profitable-but-mutually-exclusive edges. The replay counts trades skipped for capacity
(`max_positions` full), a symbol already having an open position, or a computed size of zero shares,
alongside final equity, total return %, max drawdown %, and daily Sharpe. The "Portfolio replay of PASS
lanes" section of the Markdown report renders this summary; the monthly equity curve is also written to
`research/reports/strategy_hunt/<date>T<HHMM>-equity-curve.csv`.

## Timing

From the Task 23 smoke run on `NVDA,TSLA,AMD` (2026-09-25, all ten families, warm Postgres cache):

| Step | Wall-clock | Notes |
|---|---|---|
| `export --symbols NVDA,TSLA,AMD` | 12s | SPY +103,569, VIX +0 (not in Postgres `minute_bars`), NVDA +112,153, TSLA +112,163, AMD +112,149 bars |
| `context --symbols NVDA,TSLA,AMD` | 6s | SPY +8,590, VIX +1,463 (yfinance fallback since Postgres had none — confirmed `data/strategy_hunt/daily/VIX.parquet` exists after); splits NVDA=6, TSLA=2, AMD=4 |
| `news --symbols NVDA,TSLA,AMD` | 41s | NVDA +19,015, TSLA +29,632, AMD +7,424 headlines |
| `earnings --symbols NVDA,TSLA,AMD` | 2s | 12 dates per symbol |
| `calibrate` | 19s | n=108 realized fills, median_frac=0.000127, usable=True |
| `run --symbols NVDA,TSLA,AMD --workers 3` | 24s wall (3 in parallel) | per-symbol (sequential re-run for isolated timing): NVDA 18.5s / 1,081 trades, TSLA 23.0s / 1,230 trades, AMD 12.7s / 1,023 trades |
| `score --no-coldstart` | 1.4s | verdicts: FAIL=43, WATCH=1, PASS=0 (three-symbol sample; too small for any combo to clear the n≥150 PASS floor on most families) |

Full-universe estimate (superseded by measured run below): at ~18s/symbol average for `run` across all ten families and ~500 symbols in
the universe, `run --workers 12` is roughly `500 * 18 / 12 ≈ 750s` (~13 minutes), well under the
12-hour budget noted for Task 21. `export` and `news` are bound by Postgres/Alpaca round-trips rather
than CPU; the 12s/41s above are for three symbols against a local, low-latency Postgres instance and
should not be extrapolated linearly to the full ~500-symbol universe over a 5-year window — the
Task 21 plan budgets 10–30 minutes for that full export, which this smoke run does not contradict but
also does not confirm at scale.

**Full-universe run (2026-09-25)** (measured):

| Step | Wall-clock | Notes |
|---|---|---|
| `export` | ~12 min | 539 symbols (min 250 sessions, ADV ≥ $20M) from Postgres 5-min bars |
| `news` | ~5 min | 539 symbols with four parallel fetchers (incremental cache) |
| `context` + `earnings` + `calibrate` | <1 min | regime inputs, split detection, earnings dates, fill-slippage calibration |
| `run --workers 12` | 281s | 536 symbols completed, 3 skipped (checkpoint); all ten families; measured on 12-core host |
| `run` control (50-symbol) | 31–38s | each control run for structure-destroying validation |
| `score` | 30s | verdicts + fit snapshot + report generation |
| **Total wall-clock** | **~25 min** | from `export` start to final score and report |

**Playbook Desk B1 run (2026-09-25, 13 families, FinBERT news, measured)** — full report:
`research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1.md`:

| Step | Wall-clock | Notes |
|---|---|---|
| `export --min-days 250` | 138s | incremental on a warm cache (539 symbols, ~1 new bar each) |
| `earnings` | 51s | 539 symbols, now including EPS surprises |
| `news` (incremental fetch) | 25s | ~309 new rows, FinBERT-scored at fetch time |
| `news --rescore` | 761s (12m41s) | 725,484 rows re-scored with FinBERT on one GPU, 0 errors |
| `context` | 215s | splits + universe context (458,182 rows × 539 symbols, 36 MB Parquet) |
| `calibrate` | 29s | n=108 realized fills, median_frac 0.000127, usable |
| `run --workers 12 --force` | 317s | 539 symbols, 0 errors, 13 families, 340,211 out-of-sample trades |
| controls (shuffle, cost 0.5×, cost 2×) | 36–42s each + 5s score | 50-symbol sample; each control root needs `context` symlinked too |
| `score --postgres` | 59s | 9,505 unconditional + 1,390 symbol-regime + 69 pooled rows, snapshot, report |
| `report`, `promote` | 2s each | |
| **Total wall-clock** | **~27 min** | ~13 min of it the one-off FinBERT backfill; a nightly `all` re-scores only new rows |

**Snapshot results (2026-09-25 B1 run).** 0 PASS rows anywhere: 0 of 9,505 unconditional rows (127 WATCH),
0 of 1,390 symbol-regime rows over 396 symbols (159 WATCH), 0 of 69 pooled regime rows (15 WATCH). The
strongest pooled lanes are regime-conditioned shorts in `risk_off` and longs in `risk_on`
(dt_momentum short/risk_off exp_R +0.172 on 13,370 trades, t +3.85 — WATCH, failing only `max_dd_r`).
Two cautions carried forward from the report: pooled `max_dd_r` is not scale-invariant (40 of 69 pooled rows
fail it, and only 3 of the 39 rows above the pooled `min_n` clear it), and the shuffled control produced one
pooled PASS on structure-destroyed bars where the real tape produced none — so a pooled PASS is not on its own
evidence of an edge.

**Playbook Desk B1.5 re-score (2026-09-25, hardened pooled floors, VADER news, measured)** — full report:
`research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md`; machine report
`research/reports/strategy_hunt/2026-09-25T1433-strategy-hunt.md`:

| Step | Wall-clock | Notes |
|---|---|---|
| `news --rescore` (`SENTIMENT_BACKEND=vader`) | ~5 min | `sentiment` back on VADER for live parity; `sentiment_finbert` kept, 0 errors |
| `context` | 212s | universe context rebuilt so `ctx_news_*` reflect VADER |
| `run --workers 12 --force` | 298s | 539 symbols, 0 errors, 13 families, 335,499 out-of-sample trades |
| `run --shuffle-seed 7 --workers 12 --force` | 362s | in-root baseline → `trades_shuffle/`, 539 symbols, 479,354 trades |
| control cost 2× (run + score) | 33s + 5s | 50-symbol sample in `data/strategy_hunt_cost2`; the 0.5× and separate-root shuffle controls were dropped |
| `score --postgres` | 75s | 9,231 unconditional + 1,390 symbol-regime + 69 pooled + 69 baseline rows |
| `report`, `promote` | 2s / 1s | |
| **Total wall-clock** | **~19 min** | no FinBERT backfill this time |

**Snapshot results (2026-09-25 B1.5 run).** Still 0 PASS rows anywhere: 0 of 9,231 unconditional rows
(127 WATCH), 0 of 1,390 symbol-regime rows over 396 symbols (159 WATCH), 0 of 69 pooled regime rows
(14 WATCH). The baseline guard passed — all 69 pooled rows carry a baseline covering all 539 symbols
with the same last exit date, so none is marked `no_baseline` / `baseline_mismatch` / `baseline_stale`.
The hardened floors resolved both B1 cautions. Dropping `max_dd_r` for `min_recovery` clears
dt_momentum short/risk_off (recovery 3.60 — the 639 R drawdown was a scale artefact), but the row now
fails `min_t_stat_session` (+0.56 against an iid t of +3.85) and `min_excess_r` (+0.030: the shuffled
tape reproduces +0.142 of its +0.172 expectancy). Of B1's 15 pooled WATCH rows, 14 are still WATCH
(`news_catalyst short/risk_off` fell below the WATCH `min_n` when the sentiment column went back to
VADER), 10 keep positive excess and only 3 clear the +0.05 excess floor. The closest row in the run is
`gap_play short/risk_off` — exp_R +0.181, excess +0.073, t_session +3.52, recovery 4.20 — failing
`min_decay` alone, because its last four quarters average −0.336 R. Session clustering is the binding
constraint now: 34 of the 36 pooled rows with n ≥ 500 fail `min_t_stat_session` against 29 failing the
iid `min_t_stat`.

## Operator flow

End-to-end nightly-to-promotion loop, tying together "Scheduling", "Run", and "Engine gate" above:

1. **Nightly `all`** (cron, see "Scheduling") walk-forwards every family/symbol and writes the fit
   snapshot (`data/strategy_hunt/fit/strategy_fit.json`) plus the Markdown report, then (Playbook
   Desk B2) rewrites `data/desk/playbook.json` from that snapshot's fit verdicts — see "The
   `playbook` subcommand" above and `docs/desk.md`.
2. **`scorecard --family <name>`** resolves that family's live shadow ledger
   (`data/shadow_fit_<family>.ndjson`, written by `strategies/fit_lane.py` when the family is
   registered via `FIT_LANES_ENABLED`) into simulated trades and a Stage A GO/NO_GO verdict, saved to
   `research/reports/strategy_hunt/scorecard-<family>-<date>.md`. Run this per family once its shadow
   ledger has enough history to be worth scoring — it does not require `all` to have just run.
3. **`promote`** combines the latest fit snapshot's PASS counts with each family's latest scorecard
   (re-derived the same way `scorecard` does, over every family with a ledger) and the live-facing
   flags (`Config().FIT_LANES_ENABLED`, `FIT_LANE_LIVE_ENTRIES_ENABLED`, `SWING_ALLOW_OVERNIGHT`,
   `ENABLE_STRATEGY_FIT_GATE`, `STRATEGY_FIT_SHADOW`) into one table: `pass_symbols`/`watch_symbols`
   counts, `blocker` (why it isn't promotable yet), and `recommendation` (`collect_shadow`,
   `register_shadow`, `promote_live`, or `hold`). It prints the table and the exact `.env` lines to
   apply, and writes the same content to `reports_dir(root)/strategy_hunt_promotion.md`
   (`research/reports/strategy_hunt/` for the default root). It never edits
   `.env` or `Config().yaml` itself.

   **The gate must be enabled and enforcing before any lane goes live.** `promote` reports the
   `gate_not_enforcing` blocker for every otherwise-clear family while `ENABLE_STRATEGY_FIT_GATE`
   is `False` or `STRATEGY_FIT_SHADOW` is `True` — promoting a lane to live entries while the
   engine-side per-symbol check is inert would put live orders behind no governance at all.
   Turning the gate on is therefore step zero of any promotion, not an afterthought; see the
   snapshot caveat under "Engine gate" above for what enforcing mode costs today.
4. **Apply the printed `.env` lines** by hand — e.g. add `orb` to `FIT_LANES_ENABLED` to register its
   shadow adapter, or to `FIT_LANE_LIVE_ENTRIES_ENABLED` once its scorecard is GO and it clears the
   pass-symbol floor; add `SWING_ALLOW_OVERNIGHT=1` only when `promote` flags a multiday family
   blocked by `swing_overnight_disabled`. These are comma-separated family-name lists (env-var
   precedence over yaml over code default, per `core/config.py`, search `v-strategy-fit-2026-09-25`).

   A family in `FIT_LANE_LIVE_ENTRIES_ENABLED` is still only *permitted* to trade live: the adapter
   emits a signal only for an entry whose own (symbol, family, side) row is a walk-forward `PASS`.
   `WATCH`, `FAIL`, and cold-start rows keep writing shadow-ledger lines and log
   `reason=fit_lane_shadow_verdict` instead of producing an order. So promoting a family with no
   walk-forward PASS rows for the symbols it fires on changes nothing in practice.
5. **Restart** via `./deploy.sh` so the engine picks up the new `.env` values — flags are read once at
   process start (`Config()` properties re-read `os.getenv` per call, but the fit-lane adapters
   registered in `core/engine.py` are fixed at construction time).
6. **Rollback**: remove the family from `FIT_LANE_LIVE_ENTRIES_ENABLED` (drops it back to shadow-only)
   or from `FIT_LANES_ENABLED` (drops the adapter entirely) — or unset `SWING_ALLOW_OVERNIGHT` — in
   `.env`, then restart via `./deploy.sh`. Nothing in this flow deletes shadow ledgers or fit snapshots,
   so a rolled-back family keeps accumulating shadow evidence and can be re-promoted later without
   losing history.
