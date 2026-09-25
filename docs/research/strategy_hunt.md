# Strategy hunt

Research and operations notes for the offline strategy-fit pipeline
(`research/strategy_hunt/`) and the engine-side gate that consults its
output. This file is expected to grow as later tasks in the
strategy-hunt-full-system project land (fit-lane adapters, swing
lanes, promotion tooling); for now it covers the two things Task 29
adds: the nightly scheduling of the offline hunt and the flags that
control the live engine gate.

## Scheduling

The strategy hunt (`research.strategy_hunt`) runs offline, outside the
trading engine process, and writes its snapshot to
`data/strategy_hunt/fit/strategy_fit.json` (path configurable via
`STRATEGY_FIT_PATH`). It is scheduled to run nightly after the close,
via crontab, on this host:

```
0 22 * * 1-5 cd /home/nvidia/claude/grok/trading_bot_commentary && set -a && . ./.env && set +a && /home/nvidia/anaconda3/envs/trading-bot/bin/python -m research.strategy_hunt all --workers 12 >> data/strategy_hunt/cron.log 2>&1
```

Runs 22:00 local, Monday–Friday. It sources `.env` for API
credentials, then invokes the hunt's `all` subcommand across 12
worker processes, appending stdout/stderr to
`data/strategy_hunt/cron.log`. Installing this line is an operator
step (`crontab -e`); it is not applied automatically by any code
change.

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

| Flag | Default | Purpose |
|---|---|---|
| `ENABLE_STRATEGY_FIT_GATE` | `False` | Master switch. Gate is fully inert (no audit lines, no blocking) until this is `True`. |
| `STRATEGY_FIT_SHADOW` | `True` | When the gate is enabled, run it in shadow (log the decision, never block a live entry). Must be explicitly set `False` to let the gate skip signals. |
| `STRATEGY_FIT_PATH` | `data/strategy_hunt/fit/strategy_fit.json` | Path to the snapshot file the gate and `/api/strategy-fit` read. |
| `STRATEGY_FIT_MAX_AGE_DAYS` | `7` | Snapshot age (days) beyond which it is treated as stale. |
| `FIT_LANES_ENABLED` | `""` (empty) | Comma-separated family names to register live strategy-fit lane adapters for. No lanes registered by default. |
| `FIT_LANE_LIVE_ENTRIES_ENABLED` | `""` (empty) | Comma-separated family names allowed to place LIVE entries via their fit lane; a family absent here is shadow-only even if present in `FIT_LANES_ENABLED`. |
| `SWING_ALLOW_OVERNIGHT` | `False` | Allow swing-lane positions to hold across sessions instead of forced flat by EOD. |
| `SWING_MAX_HOLD_SESSIONS` | `10` | Maximum sessions a swing-lane position may be held before a forced exit. |

All eight flags follow the config manager's env-var-then-yaml-then-default
precedence (`env ENABLE_STRATEGY_FIT_GATE=1`, yaml
`trading.enable_strategy_fit_gate: true`, etc. — see
`core/config.py`, search `v-strategy-fit-2026-09-25`).

## Run

The pipeline is a sequence of subcommands under `python -m research.strategy_hunt`, each
reading/writing under `--root` (default `data/strategy_hunt/`):

    set -a; . ./.env; set +a
    PY=/home/nvidia/anaconda3/envs/trading-bot/bin/python
    $PY -m research.strategy_hunt export --min-days 250        # Postgres minute_bars -> bars5/daily Parquet
    $PY -m research.strategy_hunt context --symbols NVDA,TSLA   # SPY/VIX regime inputs + per-symbol splits
    $PY -m research.strategy_hunt news --symbols NVDA,TSLA      # Alpaca news, incremental
    $PY -m research.strategy_hunt earnings --symbols NVDA,TSLA  # yfinance earnings dates
    $PY -m research.strategy_hunt calibrate                     # realized fill slippage from Postgres fills
    $PY -m research.strategy_hunt run --workers 12              # walk-forward per symbol; incremental, safe to rerun
    $PY -m research.strategy_hunt run --force                   # recompute everything
    $PY -m research.strategy_hunt run --shuffle-seed 7 --root data/strategy_hunt_shuffle   # structure-destroying control
    $PY -m research.strategy_hunt score --no-coldstart          # verdicts + fit snapshot + report
    $PY -m research.strategy_hunt score --root data/strategy_hunt_shuffle --no-coldstart
    cp data/strategy_hunt_shuffle/fit/strategy_fit.json data/strategy_hunt/fit/strategy_fit_shuffle.json
    $PY -m research.strategy_hunt report                        # rebuild the report from an existing fit snapshot
    $PY -m research.strategy_hunt scorecard --family orb        # Stage A shadow-resolver scorecard for one family
    $PY -m research.strategy_hunt promote                       # promotion table + .env lines for every family
    $PY -m research.strategy_hunt all --workers 12              # export -> context -> news -> earnings -> calibrate -> run -> score

Controls for cost sensitivity use `--cost-mult 0.5` / `2.0` with their own `--root`, then copy the
JSON to `fit/strategy_fit_cost_0.5.json` / `fit/strategy_fit_cost_2.0.json` before `score` so the
report picks them up. `all` is what the nightly cron job (see "Scheduling" above) runs; the other
subcommands are for ad hoc reruns of one stage (e.g. re-running `score` after editing the verdict
floors without re-simulating).

## Outputs

| Path | Content |
|---|---|
| `data/strategy_hunt/bars5/<SYM>.parquet`, `daily/`, `news/`, `earnings/`, `splits/` | Inputs (git-ignored) |
| `data/strategy_hunt/trades/<SYM>.parquet` | All out-of-sample trades for the symbol |
| `data/strategy_hunt/choices/<SYM>.json` | Chosen parameters per window and family |
| `data/strategy_hunt/checkpoints/<SYM>.json` | Skip marker for incremental runs |
| `data/strategy_hunt/calibration.json` | Realized slippage summary (`calibrate`); used as a fixed slippage override when usable |
| `data/strategy_hunt/fit/strategy_fit.json` | Latest fit snapshot (what the engine gate and `/api/strategy-fit` read) |
| `research/reports/strategy_hunt/<date>-strategy-hunt.md` | Report (verdict table, pooled metrics, portfolio replay, cost calibration) |
| `research/reports/strategy_hunt/<date>.json` | Run manifest (symbol/family counts, elapsed, errors) |
| `research/reports/strategy_hunt/<date>-equity-curve.csv` | Monthly equity curve from the portfolio replay |
| `research/reports/strategy_hunt/scorecard-<family>-<date>.md` | Stage A shadow-resolver scorecard for one family |
| `docs/research/strategy_hunt_promotion.md` | Promotion table (all families) + the `.env` lines to apply, rewritten each `promote` run |

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
`research/reports/strategy_hunt/<date>-equity-curve.csv`.

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

## Operator flow

End-to-end nightly-to-promotion loop, tying together "Scheduling", "Run", and "Engine gate" above:

1. **Nightly `all`** (cron, see "Scheduling") walk-forwards every family/symbol and writes the fit
   snapshot (`data/strategy_hunt/fit/strategy_fit.json`) plus the Markdown report.
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
   apply, and writes the same content to `docs/research/strategy_hunt_promotion.md`. It never edits
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
