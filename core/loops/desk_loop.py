"""v-playbook-desk-b2-2026-09-25 (Task 5): the desk's daily routine as a supervised loop.

Three ET wall-clock steps, once each per session:

  09:00  plan     — build the day's `DailyPlan` (regime, risk budget, max
                    trades, direction, sit-out) and persist it under
                    `<DESK_DATA_DIR>/plans/<date>.json`.
  09:35  refresh  — score the opening and attach the in-play list; when
                    `ENABLE_DESK_WATCHLIST` is on, hand that list to
                    `engine.dynamic_watchlist`.
  16:05  journal  — render the post-close journal to
                    `<DESK_DATA_DIR>/journal/<date>.md`.

Safety contract (the loop runs inside the live engine):

* Nothing here is registered unless `ENABLE_DESK` is on — see
  `engine._build_supervisor`.
* The loop only *reads* the engine (`market_state`, `dynamic_watchlist`,
  `data_provider`, `db_logger`, `_strategy_fit`) and writes exactly two things:
  desk files, and — under its own flag — the watchlist. It never places,
  cancels or blocks an order; the router hooks that consult the plan are
  separately flagged and audited.
* No step may kill the loop: every step call is wrapped, and every helper it
  uses (`core/loops/desk_steps.py`, `core/desk/*`) is written to return a
  degraded value rather than raise. A step that failed is still marked fired
  for the date, so a persistent failure logs once a day instead of every 30 s.
* Restart-safe: the fired-dates map is in memory only, and each step is
  idempotent against the file it owns — a restart at 10:00 builds today's plan
  if the file is missing, and leaves it alone if it is already there.
* Weekends are skipped (`desk_steps.is_session_day`); the repo has no holiday
  calendar, so a holiday produces an unused plan and an empty journal.

Blocking work (Parquet reads, provider REST calls, file writes) runs in
`asyncio.to_thread`, and the 09:35 scan is bounded to
`desk_steps.SCAN_CONCURRENCY` provider calls in flight.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.config import Config
from core.desk.daily_plan import build_daily_plan, load_calendar_flags, refresh_plan
from core.desk.inplay_features import load_prior_context, opening_features
from core.desk.inplay_ranker import rank_in_play
from core.desk.journal import render_journal
from core.desk.journal_data import decisions_for, trades_for, write_journal
from core.desk.models import DailyPlan, InPlayRow, PlanInputs
from core.desk.playbook import load_playbook, playbook_path
from core.desk.store import load_plan, save_plan
from core.loops import desk_steps as steps
from core.market_context import read_market_context
from core.strategy_fit_bars import load_history, merge_history

logger = logging.getLogger("TradingBot")

__all__ = ["DeskLoop"]

DEFAULT_INTERVAL_SEC = 30.0
STREAK_LOOKBACK_DAYS = 21          # calendar days of trades behind the streak rule
TRAILING_SESSIONS = 20             # journal's trailing window, in sessions
TRAILING_LOOKBACK_DAYS = 30        # calendar days that comfortably cover them
SPY = "SPY"

# (regime, spy_change_pct, vix, vix_change_pct, allows_long, allows_short)
ContextReading = Tuple[str, float, float, float, bool, bool]
_UNKNOWN_CONTEXT: ContextReading = ("unknown", float("nan"), float("nan"),
                                    float("nan"), True, True)


class DeskLoop:
    """Drives the desk's three daily steps off the ET clock.

    Dependencies are injected (config factory, clock, market-context reader) so
    the steps can be exercised against a fake engine — the real engine is not
    constructible in tests.
    """

    def __init__(self, engine: Any, interval_sec: Optional[float] = None, *,
                 config_fn: Optional[Callable[[], Any]] = None,
                 now_fn: Optional[Callable[[], datetime]] = None,
                 context_reader: Optional[Callable[[str], Any]] = None) -> None:
        self.engine = engine
        self.interval_sec = float(interval_sec) if interval_sec else DEFAULT_INTERVAL_SEC
        self._config_fn = config_fn or Config
        self._now_fn = now_fn or steps.now_et
        self._context_reader = context_reader or read_market_context
        self._fired: Dict[str, str] = {}

    # ── loop ────────────────────────────────────────────────────────

    async def run(self) -> None:
        logger.info("desk_loop started (interval=%.0fs)", self.interval_sec)
        while getattr(self.engine, "is_running", False):
            try:
                await self.tick()
            except Exception as exc:                      # pragma: no cover - tick guards itself
                logger.warning("desk_loop tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(self.interval_sec)
        logger.info("desk_loop stopped")

    async def tick(self, now: Optional[datetime] = None) -> None:
        """One 30-second pass: run whatever the clock has crossed, at most once
        per date per step. Never raises."""
        try:
            now = now if now is not None else self._now_fn()
            cfg = self._config_fn()
            schedule = self._schedule(cfg)
        except Exception as exc:
            logger.warning("desk_loop tick setup failed: %s", exc, exc_info=True)
            return

        if not steps.is_session_day(now):
            return
        date = steps.session_date_str(now)
        if not date:
            return

        for name, target, step in schedule:
            if self._fired.get(name) == date or not steps.crossed(now, target):
                continue
            try:
                await step(cfg, now, date)
            except Exception as exc:
                logger.warning("desk_loop step failed step=%s date=%s err=%s",
                               name, date, exc, exc_info=True)
            # Marked fired even on failure: one log line a day beats a retry
            # storm every 30 s against a broken dependency.
            self._fired[name] = date

    def _schedule(self, cfg: Any) -> List[Tuple[str, Tuple[int, int], Callable]]:
        return [
            ("plan", steps.parse_et_time(cfg.DESK_PLAN_TIME_ET, "09:00"), self._step_plan),
            ("refresh", steps.parse_et_time(cfg.DESK_REFRESH_TIME_ET, "09:35"),
             self._step_refresh),
            ("journal", steps.parse_et_time(cfg.DESK_JOURNAL_TIME_ET, "16:05"),
             self._step_journal),
        ]

    def _audit(self, component: str, action: str, reason: str, **fields: Any) -> None:
        """Emit one `engine_decision` line. A missing or broken `_audit` on the
        engine must not fail the step."""
        try:
            self.engine._audit(component, None, action, reason, **fields)
        except Exception as exc:
            logger.debug("desk_loop audit failed component=%s err=%s", component, exc)

    # ── 09:00 — the plan ────────────────────────────────────────────

    async def _step_plan(self, cfg: Any, now: datetime, date: str) -> None:
        existing = await asyncio.to_thread(load_plan, cfg.DESK_DATA_DIR, date)
        if existing is not None and existing.built_at:
            logger.info("desk_loop plan already built for %s — keeping it", date)
            return

        regime, spy_change, vix, vix_change, allows_long, allows_short = self._context()
        breadth = steps.breadth_score((getattr(self.engine, "market_state", None) or {})
                                      .get("breadth"))
        flags = await asyncio.to_thread(load_calendar_flags, cfg.DESK_CALENDAR_PATH, date)
        streak, prev_r = await self._recent_r(now, date)

        inputs = PlanInputs(
            date=date,
            regime=regime,
            spy_change_pct=spy_change,
            vix=vix,
            vix_change_pct=vix_change,
            breadth_score=breadth,
            calendar_flags=flags,
            losing_day_streak=streak,
            prev_day_pnl_r=prev_r,
            fit_stale=self._fit_stale(),
            gate_enforcing=bool(cfg.ENABLE_STRATEGY_FIT_GATE
                                and not cfg.STRATEGY_FIT_SHADOW),
            max_daily_loss_pct=steps.daily_loss_pct(getattr(cfg, "MAX_DAILY_LOSS", 0.0)),
            allows_long=allows_long,
            allows_short=allows_short,
        )
        plan = build_daily_plan(inputs, now)
        path = await asyncio.to_thread(save_plan, plan, cfg.DESK_DATA_DIR)

        self._audit("desk_plan", "build", "plan_built", date=date, regime=plan.regime,
                    sit_out=plan.sit_out, sit_out_reason=plan.sit_out_reason or "-",
                    max_trades=plan.max_trades,
                    risk_budget_pct=round(plan.risk_budget_pct, 3),
                    long_ok=plan.long_ok, short_ok=plan.short_ok,
                    breadth=None if breadth is None else round(breadth, 3),
                    calendar=len(flags), losing_day_streak=streak,
                    prev_day_r=round(prev_r, 3), path=path)

    def _context(self) -> ContextReading:
        """The market-context reading the plan rules need, or the unknown one.

        `read_market_context` is documented as fail-open, but it reads a cache
        the engine owns — a failure here must still leave a plan, so the desk
        falls back to "unknown regime, direction left to the plan rules".
        """
        try:
            ctx = self._context_reader(SPY)
            return (
                str(getattr(ctx, "regime", "unknown") or "unknown"),
                float(getattr(ctx, "spy_change_pct", float("nan"))),
                float(getattr(ctx, "vix_level", float("nan"))),
                float(getattr(ctx, "vix_change_pct", float("nan"))),
                bool(getattr(ctx, "allows_long", True)),
                bool(getattr(ctx, "allows_short", True)),
            )
        except Exception as exc:
            logger.warning("desk_loop market context unreadable: %s", exc)
            return _UNKNOWN_CONTEXT

    def _fit_stale(self) -> bool:
        """Is the strategy-fit snapshot stale? Unknown counts as fresh — the
        sit-out rule must fire on evidence, not on a missing reader."""
        fit = getattr(self.engine, "_strategy_fit", None)
        try:
            return bool(fit.is_stale()) if fit is not None else False
        except Exception as exc:
            logger.debug("desk_loop fit staleness unreadable: %s", exc)
            return False

    async def _recent_r(self, now: datetime, date: str) -> Tuple[int, float]:
        start = (now.date() - timedelta(days=STREAK_LOOKBACK_DAYS)).isoformat()
        trades = await trades_for((start, date), getattr(self.engine, "db_logger", None))
        return steps.daily_r_stats(trades, date)

    # ── 09:35 — the in-play list ────────────────────────────────────

    async def _step_refresh(self, cfg: Any, now: datetime, date: str) -> None:
        plan = await asyncio.to_thread(load_plan, cfg.DESK_DATA_DIR, date)
        if plan is None:
            logger.warning("desk_loop refresh skipped: no plan for %s", date)
            return

        pinned = self._pinned(cfg)
        universe = await asyncio.to_thread(steps.load_scan_universe,
                                           cfg.STRATEGY_HUNT_DATA_ROOT,
                                           cfg.DESK_SCAN_MAX_SYMBOLS)
        symbols = steps.scan_symbols(pinned, getattr(self.engine, "dynamic_watchlist", []),
                                     universe, cfg.DESK_SCAN_MAX_SYMBOLS)
        rows = await self._scan(cfg, symbols, now)
        ranked = rank_in_play(rows, pinned, cfg.DESK_IN_PLAY_N)

        refreshed = refresh_plan(plan, ranked, now)
        path = await asyncio.to_thread(save_plan, refreshed, cfg.DESK_DATA_DIR)
        self._audit("desk_inplay", "rank", "in_play_ranked", date=date, n=len(ranked),
                    scanned=len(symbols),
                    top=",".join(r.symbol for r in ranked[:5]) or "-", path=path)

        if getattr(cfg, "ENABLE_DESK_WATCHLIST", False) and ranked:
            handover = self._watchlist_from(ranked, pinned,
                                            getattr(cfg, "WATCHLIST_SIZE", None))
            self.engine.dynamic_watchlist = handover
            logger.info("desk_loop watchlist set from in-play list "
                        "(%d of %d ranked symbols)", len(handover), len(ranked))

    @staticmethod
    def _watchlist_from(ranked: Sequence[InPlayRow], pinned: Sequence[str],
                        limit: Any) -> List[str]:
        """The symbols to hand `engine.dynamic_watchlist`: pinned first, then the
        ranked in-play list, capped at `Config().WATCHLIST_SIZE`.

        v-playbook-desk-b2-review1-2026-09-25 (Minor 5): the in-play list can be
        longer than the watchlist the rest of the engine is sized for
        (`DESK_IN_PLAY_N` plus every pinned symbol), and the screener's own
        builder caps at WATCHLIST_SIZE — the desk must respect the same ceiling.
        Pinned symbols come first for the same reason they do in
        `ScreenerLoop._build_pinned_watchlist`: they are the liquid core the bot
        must keep watching. An unusable cap leaves the list uncapped rather than
        silently emptying the watchlist. The saved plan always keeps the full
        ranked list — this only shapes what the engine tracks.
        """
        order = [str(s).strip().upper() for s in (pinned or []) if str(s).strip()]
        symbols = [s for s in order if s in {r.symbol for r in ranked}]
        symbols += [r.symbol for r in ranked if r.symbol not in symbols]
        try:
            cap = int(limit)
        except (TypeError, ValueError):
            logger.debug("desk_loop unusable WATCHLIST_SIZE %r — handing over %d symbols",
                         limit, len(symbols))
            return symbols
        return symbols[:cap] if cap > 0 else symbols

    @staticmethod
    def _pinned(cfg: Any) -> List[str]:
        if not getattr(cfg, "ENABLE_PINNED_WATCHLIST", False):
            return []
        pinned = getattr(cfg, "PINNED_WATCHLIST", None)
        return [str(s).strip().upper() for s in (pinned or []) if str(s).strip()]

    async def _scan(self, cfg: Any, symbols: Sequence[str],
                    now: datetime) -> List[InPlayRow]:
        if not symbols:
            return []
        gate = asyncio.Semaphore(steps.SCAN_CONCURRENCY)
        context_cache: Dict[str, Any] = {}

        async def one(symbol: str) -> InPlayRow:
            async with gate:
                return await self._row_for(cfg, symbol, now, context_cache)

        results = await asyncio.gather(*(one(s) for s in symbols),
                                       return_exceptions=True)
        rows: List[InPlayRow] = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, InPlayRow):
                rows.append(result)
            else:
                logger.debug("desk_loop scan dropped %s: %r", symbol, result)
        return rows

    async def _row_for(self, cfg: Any, symbol: str, now: datetime,
                       context_cache: Dict[str, Any]) -> InPlayRow:
        """One candidate's opening features. Never raises — a symbol the
        provider cannot serve comes back as an all-missing row so the desk can
        see the hole rather than silently shortening its list."""
        root = getattr(cfg, "STRATEGY_HUNT_DATA_ROOT", "")
        try:
            today = await asyncio.to_thread(steps.fetch_today_bars5,
                                            getattr(self.engine, "data_provider", None),
                                            symbol)
            hist5, _daily = await asyncio.to_thread(load_history, root, symbol)
            if symbol not in context_cache:
                context_cache[symbol] = await asyncio.to_thread(load_prior_context,
                                                                root, symbol)
            history = merge_history(hist5, today)
            return opening_features(symbol, today, history, context_cache[symbol], now)
        except Exception as exc:
            logger.debug("desk_loop features failed for %s: %s", symbol, exc)
            return InPlayRow(symbol=str(symbol).strip().upper(),
                             reasons=("missing:today_bars", "error:scan"))

    # ── 16:05 — the journal ─────────────────────────────────────────

    async def _step_journal(self, cfg: Any, now: datetime, date: str) -> None:
        desk_dir = cfg.DESK_DATA_DIR
        db_logger = getattr(self.engine, "db_logger", None)

        plan = await asyncio.to_thread(load_plan, desk_dir, date)
        decisions = await decisions_for(date, db_logger)
        trades = await trades_for((date, date), db_logger)

        start = (now.date() - timedelta(days=TRAILING_LOOKBACK_DAYS)).isoformat()
        trailing_trades = await trades_for((start, date), db_logger)
        plans = await asyncio.to_thread(self._trailing_plans, desk_dir, now)
        trailing = steps.trailing_summary(trailing_trades, plans)

        book = await asyncio.to_thread(load_playbook, playbook_path(desk_dir))
        changes = steps.playbook_changes_today(book, date)

        text = render_journal(plan, decisions, trades, changes, trailing)
        path = await asyncio.to_thread(write_journal, text, desk_dir, date)
        self._audit("desk_journal", "write", "journal_written", date=date,
                    path=path or "-", chars=len(text), trades=len(trades),
                    decisions=len(decisions), playbook_changes=len(changes))

    @staticmethod
    def _trailing_plans(desk_dir: str, now: datetime) -> Dict[str, DailyPlan]:
        """The last `TRAILING_SESSIONS` weekdays' plans that exist on disk."""
        out: Dict[str, DailyPlan] = {}
        day = now.date()
        seen = 0
        while seen < TRAILING_SESSIONS:
            if day.weekday() < 5:
                seen += 1
                key = day.isoformat()
                plan = load_plan(desk_dir, key)
                if plan is not None:
                    out[key] = plan
            day -= timedelta(days=1)
        return out
