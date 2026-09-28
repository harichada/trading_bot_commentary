# The Desk (shadow)

Playbook Desk B2 (2026-09-25). Gives the bot a human trader's daily routine —
a pre-market plan, a 09:35 in-play stock ranking, an adaptive playbook keyed
per `(family, side, regime)`, and a post-close journal. Originally
shadow-only (audited, never blocking, never sizing a live order); as of the
2026-09-28 go-live restart `ENABLE_DESK_LIMITS=1` and `ENABLE_PLAYBOOK_GATE=1`
in `.env` make the `desk_limit` and `playbook` router hooks **enforcing** —
see "The four flags" § "Go-live note" below for exactly what that means and
does not mean. See the plan header for the architecture summary:
`docs/superpowers/plans/2026-09-25-playbook-desk-b2.md`. Research status —
why no lane has a walk-forward PASS yet, which is a separate question from
whether the desk's own gates enforce — is under "How a lane would be
promoted from shadow" § "Today: NO-GO" below.

## What it does, in human terms

Once a day the desk behaves like a trader arriving before the open:

1. **09:00 ET — build the plan.** It reads the SPY/VIX regime, market
   breadth, today's economic calendar, the last few sessions' P&L (looking
   for a losing streak), and whether the strategy-fit snapshot is stale
   while its gate is enforcing. From that it decides: sit out today or not
   (and why), how much of the daily loss budget it is willing to risk, how
   many trades it will allow, and whether longs, shorts, or both are in
   play. It writes this as the day's `DailyPlan`.
2. **09:35 ET — rank what's in play.** Once the opening range has printed,
   it scores a candidate list (pinned symbols, whatever the screener is
   already watching, plus the prior session's most-active symbols from the
   research universe) on opening relative volume, overnight gap, whether
   there's a catalyst (earnings or a burst of news), and 20-day relative
   strength. It keeps the top `DESK_IN_PLAY_N` and attaches them to the
   plan.
3. **16:05 ET — write the journal.** It renders a markdown report of the
   day: the plan, every governed decision (grouped by component/action,
   every skip and shadow-block spelled out), the day's closed trades,
   expectancy by family × regime for today and the trailing 20 sessions,
   any playbook status changes, and a "What to change" section derived from
   that trailing data.

Every desk decision is written to the log as an `engine_decision` line (see
"Audit lines" below), so nothing it decides is invisible even while it
changes nothing.

Overnight, a separate step — the nightly research pipeline's `playbook`
subcommand (see `docs/research/strategy_hunt.md`) — reads every resolved
shadow and live trade and rewrites the adaptive playbook: which
`(family, side, regime)` keys are trading well, which are cold, and which
have gone bad enough to suspend.

## The four flags

All four live in `core/config.py`, section `v-playbook-desk-b2-2026-09-25`,
and all default **off**. Set via env var or `Config().yaml`
(`trading.<lower_snake_case>`), same precedence as the rest of the repo's
feature flags (env wins over yaml wins over the code default).

| Flag | Default | What it enables |
|---|---|---|
| `ENABLE_DESK` | `False` | Master switch. `False` → `desk_loop` is not registered in the supervisor and neither router hook runs at all — zero behavior change. `True` → the desk builds its plan, ranks in-play stocks and writes the journal; its own sub-gates below still only *audit* unless their own flags are also on. |
| `ENABLE_DESK_WATCHLIST` | `False` | Lets the 09:35 in-play list drive `engine.dynamic_watchlist` instead of the screener's own output (the screener stops overwriting the watchlist once the plan has in-play rows; `core/loops/screener_loop.py::_desk_owns_watchlist`). Off by default: the screener keeps ownership. |
| `ENABLE_DESK_LIMITS` | `False` (**`1` in `.env` since the 2026-09-28 go-live restart — enforcing**) | Turns the daily-plan's sit-out / max-trades / direction-bias checks from "audit only" (`action=shadow`) into "actually block" (`action=skip`) at the `desk_limit` router hook. |
| `ENABLE_PLAYBOOK_GATE` | `False` (**`1` in `.env` since the 2026-09-28 go-live restart — enforcing**) | Turns a `suspended` `(family, side, regime)` key from "audit only" into "actually skip" at the `playbook` router hook, **and** writes the playbook's size multiplier into `signal.reasoning["strategy_size_multiplier"]` so `risk/manager.py` applies it. Off: the multiplier is never written (sizing is unaffected) and a suspended key is only logged. |

Both `ENABLE_DESK_LIMITS` and `ENABLE_PLAYBOOK_GATE` require `ENABLE_DESK`
to be on as well — the hook bodies are the first thing gated on `ENABLE_DESK`
in `core/engine.py`'s signal router, so a sub-flag left on with the master
switch off does nothing. (`ENABLE_DESK` itself must also be on for the go-live
restart's `ENABLE_DESK_LIMITS`/`ENABLE_PLAYBOOK_GATE` to mean anything — see
"Day-one checklist" below.)

**Go-live note (2026-09-28).** The $5,000-allocation go-live restart
(`docs/BOT_OPERATIONS_MANUAL.md` § "Live allocation and the sleeve lanes")
sets both `ENABLE_DESK_LIMITS=1` and `ENABLE_PLAYBOOK_GATE=1` — these two
gates now actually block and resize, on every lane the desk governs, not
just the new `mr_20d`/`mom_12_1` sleeve. This does **not** by itself put any
lane live: which strategies may place live orders is decided by
`FIT_LANE_LIVE_ENTRIES_ENABLED` / `FIT_LANE_LIVE_ALLOWLIST`
(`core/config.py` § `v-go-live-5k-2026-09-28`), a separate mechanism from the
desk's own governance layer (see "How a lane would be promoted from shadow"
below, which is otherwise unchanged by this restart).

**Other desk config** (same section, all with a working default — nothing
below needs to be set to run the desk):

| Property | Default | Purpose |
|---|---|---|
| `DESK_IN_PLAY_N` | `20` | How many ranked stocks the 09:35 refresh keeps. |
| `DESK_SCAN_MAX_SYMBOLS` | `60` | Cap on the *universe* slice of the 09:35 scan (pinned symbols and the live watchlist are never capped — see "Known limitations"). |
| `DESK_DATA_DIR` | `data/desk` | Root of every desk runtime artifact (git-ignored). |
| `DESK_PLAN_TIME_ET` | `09:00` | ET wall-clock time of the plan-build step. |
| `DESK_REFRESH_TIME_ET` | `09:35` | ET wall-clock time of the in-play refresh. |
| `DESK_JOURNAL_TIME_ET` | `16:05` | ET wall-clock time of the journal write. |
| `DESK_CALENDAR_PATH` | `data/desk/calendar.json` | Economic-calendar file the plan's sit-out rule reads. |
| `DESK_STREAK_SINCE` | `""` (**`2026-09-28` in `.env` since the go-live streak reset**) | Floors the `two_losing_days` sit-out rule's lookback (`STREAK_LOOKBACK_DAYS=21` by default, `core/loops/desk_loop.py`) at this ISO date — sessions before it never count as losing days. Empty (default) leaves the normal 21-day lookback in place. Set at the user's request at go-live ("start counting from today"); unparsable values are logged and ignored (`desk_loop: DESK_STREAK_SINCE unusable`), falling back to the normal window. |
| `BLOCK_NEW_ENTRIES` | `False` | **Yaml-only** — `trading.block_new_entries`, no env override by design (see `docs/BOT_OPERATIONS_MANUAL.md` § "Risk Console"). The Risk Console's immediate kill switch for new entries; exits and position management are unaffected. Set via `POST /api/console/block-entries` (un-blocking requires `confirm: "YES"`), not `.env`. Skipped signals log `engine_decision component=signal_router symbol=<SYM> action=skip reason=entries_blocked_by_console`. |
| `SETTINGS_PROMOTE_TIME_ET` | `09:25` | ET wall-clock time the desk loop promotes queued Risk Console "loosen" changes from `<DESK_DATA_DIR>/pending_settings.json` into yaml (`Config().SETTINGS_PROMOTE_TIME_ET`, yaml `trading.settings_promote_time_et` — no env override, same reasoning as `BLOCK_NEW_ENTRIES`). A "tighten" console change applies immediately and never queues. |

**Day one recommendation: `DESK_SCAN_MAX_SYMBOLS=20`.** The default 60 makes the
09:35 refresh fire roughly 60-80 provider calls in a two-minute window right
after the open (see "Known limitations"), on the same Schwab credentials the
live bot is trading on. Start the first shadow session at 20 — the in-play list
is capped at `DESK_IN_PLAY_N=20` anyway — read the `desk_inplay` audit line's
`scanned=` field, and raise it only once the rate-limit behaviour is known.

### Shadow semantics, precisely

- `ENABLE_DESK` off: nothing new runs, no new audit lines, byte-identical
  behavior to before B2.
- `ENABLE_DESK` on, both sub-flags off (today's actual posture once
  enabled — Task 8): the plan is built, in-play is ranked, the journal is
  written, and both router hooks emit `engine_decision` audit lines with
  `action=shadow` wherever they would have blocked or resized — but every
  signal proceeds exactly as it would have without the desk.
- `ENABLE_DESK_LIMITS` on: a sit-out day, a `max_trades` breach, or a
  direction the plan disallows now resolves to `action=skip` and the
  signal is dropped, at the `desk_limit` hook (immediately **before** the
  strategy-fit gate).
- `ENABLE_PLAYBOOK_GATE` on: a `suspended` key resolves to `action=skip` at
  the `playbook` hook (immediately **after** the strategy-fit gate), and
  every governed signal's size is scaled by that key's `size_mult`
  (`probation` 0.5×, `enabled` 1.0× or 1.25×, `suspended` 0.0× — moot, since
  suspended is already skipped) via `risk/manager.py::playbook_size_multiplier`,
  clamped to `[0.0, 1.5]`.

No lane gets live entries from any of this today — see "Today: NO-GO" under
"How a lane would be promoted from shadow" below.

## Timeline

Three ET wall-clock crossings, checked every 30 seconds by `DeskLoop.tick()`
(`core/loops/desk_loop.py`), each fired at most once per calendar date:

| Time (ET, default) | Step | What happens |
|---|---|---|
| 09:00 | plan | `PlanInputs` gathered (market context, breadth, calendar, losing-day streak, prior-day R, fit staleness) → `build_daily_plan` → `<DESK_DATA_DIR>/plans/<date>.json` written. A restart before this fires builds the plan on the next tick; a restart after it leaves the existing file alone. |
| 09:35 | refresh | Candidates scanned (pinned ∪ live watchlist ∪ top `DESK_SCAN_MAX_SYMBOLS` of the prior session's most-active universe symbols), opening features computed from 5-minute bars, `rank_in_play` → `refresh_plan` → the same plan file rewritten with `in_play` populated. When `ENABLE_DESK_WATCHLIST` is on, `engine.dynamic_watchlist` is set from this list. |
| 16:05 | journal | Today's decisions and trades, the trailing 20-session summary, and today's playbook changes → `render_journal` → `<DESK_DATA_DIR>/journal/<date>.md`. |

Weekends are skipped (`core.loops.desk_steps.is_session_day`, a plain
weekday check — see "Known limitations" for the holiday caveat). Every step
is wrapped so a failure logs once (`desk_loop step failed step=… date=…
err=…`) and is still marked fired for the date, rather than retrying every
30 seconds.

## Files under `data/desk/`

All git-ignored runtime output, rooted at `Config().DESK_DATA_DIR` (default
`data/desk`).

- **`plans/<date>.json`** — one `DailyPlan` per session
  (`core/desk/store.py::save_plan`/`load_plan`, atomic write — tmp file +
  `os.replace`). Shape: `date, regime, spy_change_pct, vix, vix_change_pct,
  breadth_score, calendar_flags[], long_ok, short_ok, risk_budget_pct,
  max_trades, sit_out, sit_out_reason, in_play[], notes[], built_at,
  refreshed_at`. Each `in_play` row: `symbol, score, rvol_open15, gap_pct,
  catalyst, rs_20, reasons[]`.
- **`playbook.json`** — the adaptive playbook, written nightly by
  `research.strategy_hunt playbook` (never by the live bot — the live path
  only reads it). Shape: `{"updated_at": ..., "entries": {"<family>|<side>|<regime>":
  {family, side, regime, status, size_mult, exp_r_recent, t_stat_recent,
  n_recent, updated_at, reason}}}`.
- **`journal/<date>.md`** — the post-close report (`core/desk/journal.py::render_journal`
  via `core/desk/journal_data.py::write_journal`, same atomic-write pattern).
- **`calendar.json`** (path configurable, default `data/desk/calendar.json`)
  — hand-maintained economic calendar the plan's sit-out rule reads. A JSON
  list of rows:

  ```json
  [
    {"date": "2026-09-25", "time_et": "08:30", "event": "CPI", "impact": "high"},
    {"date": "2026-09-25", "time_et": "14:00", "event": "FOMC Minutes", "impact": "medium"}
  ]
  ```

  Each row for today becomes a flag string `"<impact>:<HH:MM>:<event>"`
  (impact lowercased); a `high:` flag scheduled before 10:30 ET triggers the
  `calendar_high_impact_before_1030` sit-out rule. A missing file means "no
  scheduled events" — not an error. Malformed rows are skipped and logged,
  never raise.
- **`pending_settings.json`** — the Risk Console's queue of "loosen" changes
  awaiting the next `SETTINGS_PROMOTE_TIME_ET` promotion (`core/settings_console_store.py`,
  atomic write, same tmp-file + `os.replace` pattern). A JSON list of rows:
  `id, key, old, new, submitted_at, effective_at, actor, reason`. A "tighten"
  change never appears here — it applies immediately instead. See
  `docs/BOT_OPERATIONS_MANUAL.md` § "Risk Console".
- **`settings_audit.ndjson`** — the Risk Console's append-only audit trail,
  one JSON object per line: `ts, actor, event, key, old, new, direction,
  effective_at, reason`. `event` is one of `applied, pending, promoted,
  cancelled, rejected, block_entries, unblock_entries`. Read via
  `GET /api/console/audit` or `core/settings_console_store.py::read_audit`
  (never raises; a malformed line is skipped and logged, not fatal).

## Audit lines to grep

Every desk decision emits one `engine_decision` line per governed signal (or
per loop step). Components, in router order:

- `desk_limit` — the daily-plan's sit-out/max-trades/direction check,
  emitted **before** the `strategy_fit` gate. `reason` is one of
  `desk_sit_out:<reason>`, `desk_max_trades`, `desk_direction_long_off`,
  `desk_direction_short_off`, or `desk_ok` (an explicit allow is audited
  too, so the journal's counts see the full trail).
- `strategy_fit` — unchanged from Playbook Desk B1 (see
  `docs/BOT_OPERATIONS_MANUAL.md` § "Strategy fit gate").
- `playbook` — the adaptive playbook check, emitted **immediately after**
  the strategy-fit gate. `reason` is `no_playbook_entry`, `playbook_suspended`,
  `playbook_probation`, or `playbook_enabled`; carries `size_mult`, `status`,
  and (when it fell back from an unmeasured exact regime)
  `playbook_fallback=mixed|unknown`.
- `desk_plan` — the 09:00 loop step. `action=build`, fields `date, regime,
  sit_out, sit_out_reason, max_trades, risk_budget_pct, long_ok, short_ok,
  breadth, calendar, losing_day_streak, prev_day_r, path`.
- `desk_inplay` — the 09:35 loop step. `action=rank`, fields `date, n,
  scanned, top` (top 5 symbols), `path`.
- `desk_journal` — the 16:05 loop step. `action=write`, fields `date, path,
  chars, trades, decisions, playbook_changes`.

Example grep:

```bash
grep 'engine_decision component=desk_' trading_bot.log | jq -r .message
grep 'engine_decision component=playbook' trading_bot.log | jq -r .message
```

`action` on the router-hook components (`desk_limit`, `playbook`) is one of
`allow` / `shadow` / `skip` — `shadow` means "this would have blocked, but
its flag is off, so the signal proceeded"; `skip` means it actually blocked.

## Endpoints

All three are read-only, file-backed (no engine-state dependency), and
never return an HTTP 500 — any failure degrades to `{"ok": false, "error":
...}` with a 200, the same contract as `GET /api/strategy-fit`
(`api/routes.py`). `date` query params default to today (ET) and are
rejected (`{"ok": false, "error": "invalid_date"}`) unless they match
`YYYY-MM-DD` and parse as a real calendar date.

**`GET /api/desk/plan?date=YYYY-MM-DD`** — today's (or `date`'s) plan.

```json
{
  "ok": true,
  "date": "2026-09-25",
  "plan": {
    "date": "2026-09-25", "regime": "risk_on", "spy_change_pct": 0.42,
    "vix": 14.8, "vix_change_pct": -1.2, "breadth_score": 0.58,
    "calendar_flags": ["high:08:30:CPI"],
    "long_ok": true, "short_ok": false,
    "risk_budget_pct": 5.0, "max_trades": 6,
    "sit_out": false, "sit_out_reason": "",
    "in_play": [
      {"symbol": "NVDA", "score": 0.71, "rvol_open15": 2.4, "gap_pct": 1.9,
       "catalyst": 1.0, "rs_20": 6.2, "reasons": []}
    ],
    "notes": ["regime risk_on: risk budget 1.00x of the 5.00% daily loss cap = 5.00%",
              "max trades 6 for regime risk_on"],
    "built_at": "2026-09-25T09:00:03-04:00", "refreshed_at": "2026-09-25T09:35:11-04:00"
  }
}
```

No plan for that date: `{"ok": false, "date": "2026-09-25", "error": "no_plan"}`.

**`GET /api/desk/playbook`** — the current adaptive playbook.

```json
{
  "ok": true,
  "playbook": {
    "updated_at": "2026-09-25T22:14:07",
    "entries": {
      "mean_rev_intraday|long|risk_on": {
        "family": "mean_rev_intraday", "side": "long", "regime": "risk_on",
        "status": "probation", "size_mult": 0.5,
        "exp_r_recent": -0.08, "t_stat_recent": -0.6, "n_recent": 7,
        "updated_at": "2026-09-25T22:14:07", "reason": "probation_small_sample"
      }
    }
  }
}
```

No file yet: `{"ok": false, "error": "no_playbook"}`.

**`GET /api/desk/journal?date=YYYY-MM-DD`** — the rendered markdown journal.

```json
{"ok": true, "date": "2026-09-25", "markdown": "## Plan\n- regime: risk_on\n..."}
```

No journal for that date: `{"ok": false, "date": "2026-09-25", "error": "no_journal"}`.

`tools/bot_monitor.py`'s report carries a "## Desk" section (positioned
**before** "## Strategy fit") built from `/api/desk/plan` and
`/api/desk/playbook`: date, regime, sit-out + reason, long/short ok, risk
budget, max trades, top 5 in-play with score, and playbook status counts
(enabled/probation/suspended). It prints `_desk not enabled / no plan_` when
the plan endpoint reports `ok: false`.

## Playbook rules

`core/desk/playbook.py::update_entry(prev, recent, fit_verdict, now)` — the
nightly transition for one `(family, side, regime)` key, applied to its last
`RECENT_WINDOW=60` resolved trades (newest last). `recent_stats` computes
`n`, `exp_r` (mean R) and a trade-level t-stat (`0.0` below 2 trades; ±inf
when the sample has zero dispersion around a non-zero mean).

| Rule (first match wins) | Condition | Status | `size_mult` | `reason` |
|---|---|---|---|---|
| 1 | `n ≥ 10` and `exp_r < 0` and `t_stat < -1.0` | `suspended` | `0.0` | `suspended_negative_expectancy` |
| 2 | `exp_r < 0` (not rule 1) | `probation` | `0.5` | `probation_negative_expectancy` |
| 3 | `n < 10` | `probation` | `0.5` | `probation_small_sample` |
| 4 | `exp_r ≥ 0.15` and `n ≥ 40` and `fit_verdict ∈ {PASS, WATCH}` | `enabled` | `1.25` | `enabled_upsize_fit_confirmed` |
| 5 | otherwise | `enabled` | `1.0` | `enabled_normal` |

`size_mult` is always clamped to `[0.0, 1.5]`, including on read-back from
disk — a hand-edited `playbook.json` cannot upsize a lane past 1.5×.

**Lapse.** `carry_forward(prev, now, has_recent_trades)`: a `suspended` key
with no resolved trade in `SUSPENSION_LAPSE_DAYS=30` days lapses back to
`probation` (`reason=suspension_lapsed_no_recent_trades`) — a suspended key
places no live orders, so 30 days of silence is the expected state, not
continuing evidence against it. `enabled`/`probation` keys carry forward
unchanged when there is nothing new to measure.

**Fallback (live gate lookup, `evaluate_playbook_gate`).** Exact
`(family, side, regime)` first; if the live regime is `unknown`, fall back
to that family/side's `mixed` entry (`playbook_fallback=mixed`); otherwise
fall back to the `unknown` entry (`playbook_fallback=unknown`). A measured
regime is never governed by another measured regime's entry — only
`unknown` falls back.

## How to read a journal

`data/desk/journal/<date>.md`, sections in order:

1. **Plan** — regime, bias (`long+short` / `long only` / `short only` /
   `neutral (no bias)`), risk budget, max trades, sit-out (yes/no + reason),
   top 10 in-play with score/rvol/gap/reasons.
2. **Decisions** — every governed decision for the day, grouped
   `component/action: count`, then every `skip` line (actually blocked), then
   the `shadow` lines that would have blocked. That last list is restricted to
   the three components the desk actually governs (`desk_limit`,
   `strategy_fit`, `playbook` — `core/desk/journal.py::GOVERNED_COMPONENTS`):
   other components write `action="shadow"` for their own purposes, and
   counting those as "would have blocked" overstated the desk's effect. They
   are still in the `component/action` summary above.
3. **Trades** — symbol, side, family, R, exit reason, hold time for each
   closed trade. A round trip belongs to the session it was **closed** in:
   `DbLogger.get_bot_trades_between` filters on `DATE(exit_time)`, matching
   `core/loops/desk_trade_stats.py::_row_date`. So a position opened yesterday
   and closed today appears in today's journal, and in today's R for the
   losing-day-streak rule.
4. **Expectancy by Setup x Regime** — `family/regime: n=… exp_R=… PF=…`,
   both for today and trailing 20 sessions.
5. **Playbook Changes** — every entry the nightly updater stamped today
   (`old_status -> new_status (reason)`; `old_status` is left blank since
   the playbook file only keeps current state, not history).
6. **What to Change** — three derived rules over the trailing window: a
   family/regime with trailing `n ≥ 20` and `exp_R < 0` → "suspend
   candidate"; a sit-out day whose trades-that-would-have-happened R was
   positive → "review sit-out rule"; a day that hit its `max_trades` cap →
   "raise/lower max_trades".

Every section degrades to a `none` line on missing or malformed input —
`render_journal` never raises, so a partial upstream read (e.g. Postgres
down at 16:05) still produces a readable file.

## How a lane would be promoted from shadow

The desk's own gates (`desk_limit`, `playbook`) are a separate governance
layer from the strategy-fit gate and the fit-lane promotion flow described
in `docs/research/strategy_hunt.md` § "Operator flow" — they do not
themselves decide which strategies may trade live. The path from shadow to
something that matters:

1. **Research PASS.** A family/symbol/side needs a walk-forward `PASS`
   verdict in the nightly strategy-fit snapshot before its fit lane can
   fire live at all (`FIT_LANE_LIVE_ENTRIES_ENABLED` + a `PASS` row —
   `docs/research/strategy_hunt.md` § "Operator flow", step 4).
2. **Shadow scorecard.** Once a fit lane is registered
   (`FIT_LANES_ENABLED`) it accumulates a shadow ledger; `scorecard
   --family <name>` turns that into a Stage A GO/NO_GO verdict.
3. **Playbook evidence.** In parallel, the desk's own playbook accumulates
   `(family, side, regime)` evidence from both the shadow ledgers and any
   live trades. A key only reaches `enabled 1.25×` once it has `n ≥ 40`
   resolved trades, `exp_R ≥ 0.15`, *and* a `PASS`/`WATCH` fit verdict —
   research and the tape have to agree.
4. **Enable the desk's own gates.** Only once a lane clears both 1 and 2
   above does turning on `ENABLE_DESK_LIMITS` / `ENABLE_PLAYBOOK_GATE` mean
   anything beyond risk hygiene — until then they are defensive-only (see
   "Today: NO-GO" just below).

**Today: NO-GO.** No family has a walk-forward PASS in any regime as of the
last full-universe run (`research/reports/strategy_hunt/2026-09-25-strategy-hunt-b1-5.md`
§ B1.6: 0 of 9,227 unconditional rows, 0 of 1,625 symbol-regime rows, 0 of 67
universe-pooled regime rows). The earlier "regime split is real and large"
finding from the B1/B1.5 runs was **withdrawn** after B1.6 corrected the
regime label to entry-time information — the split had been fitted to
information the desk could not have known when the trade was taken. So the
desk's regime-keyed `risk_budget_pct`/`max_trades` rules in `core/desk/daily_plan.py`
are **defensive risk rules** (spend less, trade less, in a worse regime),
not evidence of an edge — and the playbook learns per `(family, side,
regime)` from shadow and live results precisely because no research result
yet says any one of those keys is worth trading harder than another.

## Known limitations

Carried over from `.superpowers/sdd/2026-09-25-playbook-desk-b2/task-5-report.md`
(Task 5's self-review) unless noted otherwise:

- **R on the un-migrated `bot_trades` schema is derived from the *closing*
  stop.** `scripts/migrate_ledger_integrity.py` (gated by `LEDGER_INTEGRITY`,
  default **off**) is what adds `initial_stop` / `initial_risk_per_share` /
  `pnl_r` to `bot_trades`; until it runs, the live tape's R comes from
  `core.desk.r_multiple.derive_pnl_r`, i.e. `pnl / (abs(entry_price -
  stop_loss) * quantity)` — and `stop_loss` on a closed row is the stop **as it
  stood at exit**, not the stop the trade was risked against. Two consequences,
  both of which bias the playbook that learns from those rows:
  - a stop moved to break-even makes the denominator zero, so R is `None` and
    the trade is **dropped** from the tape entirely (a winner that was trailed
    to break-even and a scratch both vanish, so the sample skews);
  - a scaled-out or trailed position measures its full P&L against a *tightened*
    stop distance, so R comes out **inflated**.
  A faithful R needs the migration (it records the initial stop at entry and
  writes `pnl_r` directly, which `derive_pnl_r` then prefers). Applying it is a
  live schema change and therefore the user's decision; until then, read the
  playbook's live-sourced `exp_R` as indicative, and prefer the shadow-ledger
  rows (which carry their own R) when the two disagree.
- **A suspension lapses on silence, and in shadow it never gets the chance.**
  `carry_forward` puts a `suspended` key back on `probation` after
  `SUSPENSION_LAPSE_DAYS=30` days with no resolved trade, because a key the gate
  is blocking cannot produce evidence in its own defence. That rule only ever
  fires with `ENABLE_PLAYBOOK_GATE` **on**: during the shadow period the gate
  blocks nothing, so a suspended key keeps trading and keeps being re-measured
  each night from its own recent results. Two things follow. In shadow, a
  suspension is only as stable as the last `RECENT_WINDOW=60` trades — expect
  keys to flip status between sessions, and read the journal's "Playbook
  Changes" section rather than any one night's file. Once the gate enforces, no
  suspension is permanent: a blocked key returns to `probation` (0.5x size)
  after 30 quiet days, trades again at half size, and is re-judged on what that
  produces.
- **`trades_today` counts symbols, not trades.** The desk's `max_trades`
  counter comes from `DbLogger.get_todays_bot_entries()`, which is keyed by
  symbol — two entries in the same symbol on the same day count once. This
  is what the plan brief prescribed as the fallback (there is no
  per-day-entry counter on the engine today).
- **Weekday-only calendar.** `is_session_day` is a plain weekday check; the
  repo has no holiday calendar (`engine.is_market_hours` is weekday-based
  too). A market holiday still builds a plan and writes an (empty) journal.
- **A failed step is not retried the same day.** Each step is marked fired
  even on failure, so a broken dependency logs once instead of every 30
  seconds — but a step that failed at 09:00 will not retry until the next
  session.
- **~60–80 provider calls at 09:35.** The in-play scan pulls a 10-day/5-minute
  bar window per candidate (pinned + watchlist + up to `DESK_SCAN_MAX_SYMBOLS`
  universe symbols), bounded to 4 concurrent requests. At the default cap of
  60 that is roughly 60–80 REST calls in ~15–20 batches right after the
  open — worth watching against provider rate limits on the first live
  shadow day (hence `DESK_SCAN_MAX_SYMBOLS=20` on day one, above). Note that
  until the whole-branch review's I1 fix, `data_providers/schwab.py` answered
  *any* `period > 1` minute request with **daily** candles; it now routes by
  `frequency_type` (`history_call_name`), so the first shadow session is also
  the first time the in-play features — and the strategy-fit lane's
  `_fit_bars_source` — actually see 5-minute bars.
- **Breadth mapping is heuristic.** Providers disagree on shape
  (`advancing_percent`, `advances`/`declines`, a signed `advance_decline`,
  `new_highs`/`new_lows`); each is mapped to a 0–1 scale by
  `core/loops/desk_steps.py::breadth_score`, most-direct reading first. Only
  the plan's `long_ok`/`short_ok` thresholds (0.40 / 0.60) consume it.
- **Neither flag creates a new order path.** `ENABLE_DESK_LIMITS` and
  `ENABLE_PLAYBOOK_GATE` only govern signals that already exist (block or
  resize them); which strategies may emit a live signal at all is still
  decided by each lane's own `*_LIVE_ENTRIES_ENABLED` flag and, for the fit
  lanes, `FIT_LANE_LIVE_ENTRIES_ENABLED`/`FIT_LANE_LIVE_ALLOWLIST`. Both
  `ENABLE_DESK_LIMITS` and `ENABLE_PLAYBOOK_GATE` **are** on in the live
  deployment as of the 2026-09-28 go-live restart (see "The four flags"
  above) — as of 13:17 ET the same day they are also governing the four
  intraday builtin lanes (`day_trade_momentum`, `day_trade_momentum_short`,
  `mean_reversion`, `orb_contraction_rvol`), not just the `mr_20d`/`mom_12_1`
  sleeve; see `docs/BOT_OPERATIONS_MANUAL.md` § "Live allocation and the
  sleeve lanes".
- **The plan's `strategies/fit_lane.py::LANE_MAP` citation does not match
  the code** (`docs/superpowers/plans/2026-09-25-playbook-desk-b2.md`,
  Global Constraints: "Family naming: `strategies/fit_lane.py::LANE_MAP`
  maps live strategy names to family names"). The live strategy → family
  map actually lives at
  `core/strategy_fit_gate.py::LANE_MAP` (read via `family_for()`); the
  research-side copy the nightly updater uses is
  `research/strategy_hunt/playbook_live.py::FAMILY_BY_STRATEGY`, pinned
  equal to it by `tests/research/strategy_hunt/test_playbook_update.py`.
  Both this file and the code comments in `core/desk/journal_data.py` and
  `research/strategy_hunt/playbook_live.py` note the correction.

## Day-one checklist (go-live, 2026-09-28)

Before the first live session with `ENABLE_DESK_LIMITS=1` /
`ENABLE_PLAYBOOK_GATE=1` enforcing. **Update, 13:17 ET the same day:** the
four intraday builtin lanes (`ORB_LIVE_ENTRIES_ENABLED`,
`MEAN_REV_LIVE_ENTRIES_ENABLED`, `DAY_TRADE_LIVE_ENTRIES_ENABLED`,
`DAY_TRADE_SHORT_LIVE_ENTRIES_ENABLED`) were also switched on, so both
enforcing flags now govern those signals too — `dt_momentum|long` is
suspended (`size_mult` 0.0, every entry skipped) and `mean_rev_intraday|long`
is on probation (`size_mult` 0.5) per `data/desk/playbook.json`; the
day-trade lanes are also still bound by their own `DAY_TRADE_FLATTEN_HOUR`
(default `15`, i.e. 3 PM ET) entry cutoff regardless of the playbook. See
`docs/BOT_OPERATIONS_MANUAL.md` § "Live allocation and the sleeve lanes" for
the full accounting:

1. **`ENABLE_DESK=1` is set.** Both enforcing sub-flags are inert without it
   (see "The four flags" above). Confirm with `GET /api/desk/plan` returning a
   populated plan rather than an empty/absent one.
2. **`DESK_SCAN_MAX_SYMBOLS`.** The default `60` drives roughly 60-80 provider
   calls in the two minutes after the 09:35 refresh, on the same Schwab
   credentials the live sleeve now trades on. Leave it at the recommended `20`
   for the first session (see "The four flags" above) unless the rate-limit
   behaviour is already known good.
3. **09:00 ET — read the plan.** `GET /api/desk/plan` or
   `grep 'engine_decision component=desk_limit' trading_bot.log`. Confirm it
   is not a sit-out day for a reason that surprises you (`two_losing_days`,
   stale strategy-fit snapshot, economic-calendar blackout, VIX spike — first
   match wins, `core/desk/daily_plan.py`). The losing-day streak only looks
   back to `DESK_STREAK_SINCE` (`.env` `2026-09-28` since the go-live reset —
   see "Other desk config" above); a `two_losing_days` sit-out from a losing
   streak that started before that date is a bug, not the rule working.
4. **09:35 ET — confirm the in-play refresh ran.** The plan file's `in_play`
   list should be populated; `scanned=` in the `desk_inplay` audit line
   should roughly match `DESK_SCAN_MAX_SYMBOLS`.
5. **Watch `component=desk_limit` and `component=playbook` for `action=skip`.**
   This is the enforcing behaviour working as intended, not an error — but a
   sit-out day or a suspended key that skips *every* signal on day one is
   worth a manual look before assuming it is correct.
6. **Confirm the sleeve is sized on the allocation, not the account.**
   Cross-check a filled order's notional against
   `docs/BOT_OPERATIONS_MANUAL.md` § "Live allocation and the sleeve lanes" §
   "Sizing table" ($800-$1,000 per position at go-live settings) — the
   dashboard's account tiles show the raw ~$46.5k/~$93k account and will not
   reflect this.
7. **16:05 ET — read the journal.** `GET /api/desk/journal` or
   `data/desk/journal/<date>.md`: every skip/shadow-block by component, the
   day's closed trades, and any playbook status changes. This is the primary
   end-of-day review artifact — read it before the next session's plan
   builds.
8. **If a lane should be paused immediately**, see
   `docs/BOT_OPERATIONS_MANUAL.md` § "How to stop trading, in one step"
   (`POST /api/stop`, or clear `FIT_LANE_LIVE_ENTRIES_ENABLED` and restart) —
   turning `ENABLE_DESK_LIMITS`/`ENABLE_PLAYBOOK_GATE` back off only changes
   whether the desk's own checks block; it does not stop new entries from
   other gates.
