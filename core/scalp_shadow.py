"""Scalp reversal shadow loop — runs the signal in paper mode during RTH.

Each tick (60 s):
  1. Load today's 1-min bars + 100-bar history from Postgres per symbol.
  2. Re-run replay_symbol_scalp on the combined window.
  3. Log any newly closed trades (not yet in the ndjson ledger).
  4. Expose open positions + today's stats via get_status().

Completed trades → data/shadow_scalp_reversal.ndjson
Current state    → data/shadow_scalp_state.json  (open positions + dedup set)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import DEFAULT_DSN, compute_indicators
from backtest.scalp_reversal import ScalpParams, allocate_slots, load_bars, replay_symbol_scalp

logger = logging.getLogger("TradingBot")

_REPO_ROOT = Path(__file__).resolve().parent.parent
NDJSON_PATH = _REPO_ROOT / "data" / "shadow_scalp_reversal.ndjson"
STATE_PATH  = _REPO_ROOT / "data" / "shadow_scalp_state.json"

# Default parameters (updated 2026-11-24: both sides + per-stock direction gate)
DEFAULT_SHADOW_PARAMS = ScalpParams(
    rsi_long_thresh=45.0,
    rsi_short_thresh=60.0,
    vol_min=1.0,
    atr_stop_mult=2.0,
    atr_target_mult=3.0,
    quick_exit_bars=10,
    min_profit_pct=0.15,
    max_hold_bars=5,
    adx_max=999.0,
    require_reversal_candle=False,
    candle_wick_ratio=1.5,
    entry_confirm=False,
    rsi_turn=False,
    sides="both",
    direction_gate=True,
    from_open_drop_max=0.0,
)

DEFAULT_SYMBOLS = [
    "NVDA", "TSLA", "QQQ", "SMCI", "IREN", "MU",
    "SNAP", "LCID", "PURR", "POET", "RIVN", "NBIS",
    "NOK", "AAL",
]

SLOT_SIZE_USD = 10_000


def _now_et() -> datetime:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("America/New_York"))


def _is_rth() -> bool:
    """True if current time is within regular trading hours (Mon–Fri 9:30–16:00 ET)."""
    now = _now_et()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes <= 16 * 60


def _today_et() -> str:
    return _now_et().strftime("%Y-%m-%d")


class ScalpShadowEngine:
    """Paper-trades the scalp reversal signal using live Postgres minute bars."""

    def __init__(
        self,
        symbols: list[str] = DEFAULT_SYMBOLS,
        params: ScalpParams = DEFAULT_SHADOW_PARAMS,
        dsn: str = DEFAULT_DSN,
        slot_size: int = SLOT_SIZE_USD,
        tick_interval: int = 60,
        max_slots: int = 2,
    ) -> None:
        self.symbols = symbols
        self.params = params
        self.dsn = dsn
        self.slot_size = slot_size
        self.tick_interval = tick_interval
        self.max_slots = max_slots

        # (symbol, side, entry_time_iso) → already logged
        self._logged: set[tuple[str, str, str]] = set()
        # (entry_time_ts, exit_time_ts) of committed today trades — used to
        # enforce the slot cap across incremental _tick calls.
        self._logged_intervals: list[tuple] = []
        # (symbol, date) → stale warning already emitted
        self._stale_warned: set[tuple[str, str]] = set()
        # symbol → open position dict (None if flat)
        self._open: dict[str, dict | None] = {s: None for s in symbols}
        # today's completed trades (in-memory cache, flushed at day start)
        self._today_trades: list[dict] = []
        self._today_date: str = ""
        # 3 consecutive losses → block symbol for the rest of the day
        self._consec_losses: dict[str, int] = {s: 0 for s in symbols}
        self._blocked_today: set[str] = set()

        self._task: asyncio.Task | None = None
        self._load_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="scalp_shadow_loop")
        logger.info("scalp_shadow started symbols=%d interval=%ds",
                    len(self.symbols), self.tick_interval)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def update_sides(self, sides: str) -> None:
        """Hot-swap the sides filter without restarting the loop."""
        if sides not in ("both", "long_only", "short_only"):
            raise ValueError(f"Invalid sides value: {sides!r}")
        # ScalpParams is frozen; replace the whole params object
        from dataclasses import replace
        self.params = replace(self.params, sides=sides)
        logger.info("scalp_shadow sides updated to %s", sides)

    def get_status(self) -> dict:
        """Return a JSON-serializable snapshot of today's shadow activity."""
        today = _today_et()
        trades = [t for t in self._today_trades if t["date"] == today]

        completed = [t for t in trades if not t.get("open")]
        wins = [t for t in completed if t.get("dollar_pnl", 0) > 0]
        total_pnl = sum(t.get("dollar_pnl", 0) for t in completed)
        win_rate = len(wins) / len(completed) * 100 if completed else 0.0

        open_pos = []
        for sym, pos in self._open.items():
            if pos is not None:
                open_pos.append({
                    "symbol": sym,
                    "side": pos["side"],
                    "entry_time": pos["entry_time"],
                    "entry_px": pos["entry_px"],
                    "stop_px": pos["stop_px"],
                    "target_px": pos["target_px"],
                    "hold_bars": pos["hold_bars"],
                })

        return {
            "date": today,
            "is_rth": _is_rth(),
            "params": asdict(self.params),
            "symbols": self.symbols,
            "blocked_today": sorted(self._blocked_today),
            "slot_size": self.slot_size,
            "max_slots": self.max_slots,
            "summary": {
                "n_trades": len(completed),
                "n_open": len(open_pos),
                "total_pnl": round(total_pnl, 2),
                "win_rate_pct": round(win_rate, 1),
                "n_wins": len(wins),
                "n_stops": sum(1 for t in completed if t.get("exit_reason") == "stop"),
            },
            "open_positions": open_pos,
            "completed_trades": sorted(completed, key=lambda t: t.get("exit_time", ""), reverse=True),
        }

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            try:
                today = _today_et()
                if today != self._today_date:
                    self._on_new_day(today)
                await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("scalp_shadow tick error: %s", exc, exc_info=True)
            await asyncio.sleep(self.tick_interval)

    def _on_new_day(self, date: str) -> None:
        self._today_date = date
        # Keep old logged keys so we never re-log a trade from yesterday
        # but clear open positions (EOD would have closed them)
        self._open = {s: None for s in self.symbols}
        self._today_trades = [t for t in self._today_trades if t["date"] == date]
        self._logged_intervals = []  # reset slot timeline for the new day
        self._consec_losses = {s: 0 for s in self.symbols}
        self._blocked_today = set()
        logger.info("scalp_shadow new day %s", date)

    def _record_loss_streak(self, symbol: str, dollar_pnl: float) -> None:
        if dollar_pnl < 0:
            self._consec_losses[symbol] = self._consec_losses.get(symbol, 0) + 1
            if self._consec_losses[symbol] >= 3:
                self._blocked_today.add(symbol)
                logger.info(
                    "scalp_shadow %s blocked for today (3 consecutive losses)",
                    symbol,
                )
        else:
            self._consec_losses[symbol] = 0

    def _tick(self) -> None:
        """Load all symbols, replay each independently, then apply global slot cap
        via allocate_slots — identical methodology to the backtest's run()."""
        today = _today_et()
        all_candidates: list = []
        loaded: dict[str, object] = {}  # symbol → DataFrame

        for symbol in self.symbols:
            if symbol in self._blocked_today:
                continue
            try:
                df = load_bars(symbol, days=5, dsn=self.dsn)
                if df.empty or len(df) < 51:
                    continue
                last_bar_date = _bar_date(df.index[-1])
                if last_bar_date != today:
                    stale_key = (symbol, today)
                    if stale_key not in self._stale_warned:
                        self._stale_warned.add(stale_key)
                        logger.warning(
                            "scalp_shadow %s: DB stale — last bar %s, today %s",
                            symbol, last_bar_date, today,
                        )
                    continue
                loaded[symbol] = df
                px_min = df["Close"].min() * 0.95
                px_max = df["Close"].max() * 1.05
                raw_candidates = replay_symbol_scalp(symbol, df, self.params)
                valid = [t for t in raw_candidates
                         if px_min <= t.entry_price <= px_max]
                if len(valid) < len(raw_candidates):
                    logger.warning(
                        "scalp_shadow %s: dropped %d out-of-range trades "
                        "(entry must be in [%.2f, %.2f])",
                        symbol, len(raw_candidates) - len(valid), px_min, px_max,
                    )
                all_candidates.extend(valid)
            except Exception as exc:
                logger.debug("scalp_shadow %s load error: %s", symbol, exc)

        # Apply global slot cap the same way the backtest does.
        executed = allocate_slots(all_candidates, self.max_slots)

        # Log any newly closed executed trades — but only if they don't
        # exceed max_slots concurrent positions among already-committed trades.
        import pandas as _pd
        for trade in executed:
            if _bar_date(trade.exit_time) != today:
                continue
            key = (trade.symbol, trade.side, _iso(trade.entry_time))
            if key in self._logged:
                continue
            entry = trade.entry_time
            if not isinstance(entry, _pd.Timestamp):
                entry = _pd.Timestamp(entry)
            exit_ = trade.exit_time
            if not isinstance(exit_, _pd.Timestamp):
                exit_ = _pd.Timestamp(exit_)
            # Count already-committed trades that overlap with this trade's window.
            overlap = sum(1 for (e, x) in self._logged_intervals if e <= entry < x)
            if overlap >= self.max_slots:
                continue  # slot cap reached — skip this trade
            self._logged.add(key)
            self._logged_intervals.append((entry, exit_))
            record = self._trade_to_record(trade, today)
            self._append_ndjson(record)
            self._today_trades.append(record)
            self._record_loss_streak(trade.symbol, record["dollar_pnl"])
            logger.info(
                "scalp_shadow logged %s %s entry=%s exit=%s pnl=%.2f",
                trade.symbol, trade.side, _iso(trade.entry_time),
                _iso(trade.exit_time), record["dollar_pnl"],
            )

        # Update open-position display for each loaded symbol.
        for symbol, df in loaded.items():
            try:
                self._update_open_from_replay(symbol, df, today)
            except Exception as exc:
                logger.debug("scalp_shadow %s open update error: %s", symbol, exc)

    def _update_open_from_replay(self, symbol: str, df: pd.DataFrame, today: str) -> None:
        """Determine if the last signal bar has an unclosed position."""
        from backtest.engine import compute_indicators
        import numpy as np
        ind = compute_indicators(df)

        if df.index.tz is None:
            et_index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            et_index = df.index.tz_convert("America/New_York")
        et_dates = et_index.normalize()

        required = {"rsi", "macd", "macd_signal", "bb_lower", "bb_middle",
                    "bb_upper", "atr", "adx", "sma_20", "sma_50",
                    "volume_ratio", "high_20", "low_20"}

        open_long: dict | None = None
        open_short: dict | None = None
        pending_long: dict | None = None
        pending_short: dict | None = None
        day_bar_count: dict = {}
        rth_open: dict = {}
        params = self.params

        for i in range(50, len(df)):
            bar = df.iloc[i]
            indicators = ind.iloc[i].to_dict()
            if any(pd.isna(indicators.get(k)) for k in required):
                continue

            bar_date = et_dates[i]
            day_bar_count[bar_date] = day_bar_count.get(bar_date, 0) + 1
            if day_bar_count[bar_date] == 1:
                pending_long = None
                pending_short = None
                rth_open[bar_date] = float(df.iloc[i]["Open"])

            is_last_bar_of_day = (
                i == len(df) - 1
                or et_dates[i + 1] != bar_date
            )

            close_px = float(bar["Close"])
            bar_ts = df.index[i]

            # --- Manage open long ---
            if open_long is not None:
                open_long["hold"] += 1
                pos = open_long
                unrealized_pct = (close_px - pos["entry"]) / pos["entry"] * 100.0
                closed = False
                if (pos["hold"] <= params.quick_exit_bars
                        and unrealized_pct >= params.min_profit_pct):
                    closed = True
                elif float(bar["Low"]) <= pos["stop"]:
                    closed = True
                elif float(bar["High"]) >= pos["target"]:
                    closed = True
                elif pos["hold"] >= params.max_hold_bars:
                    closed = True
                elif is_last_bar_of_day:
                    closed = True
                if closed:
                    open_long = None

            # --- Manage open short ---
            if open_short is not None:
                open_short["hold"] += 1
                pos = open_short
                unrealized_pct = (pos["entry"] - close_px) / pos["entry"] * 100.0
                closed = False
                if (pos["hold"] <= params.quick_exit_bars
                        and unrealized_pct >= params.min_profit_pct):
                    closed = True
                elif float(bar["High"]) >= pos["stop"]:
                    closed = True
                elif float(bar["Low"]) <= pos["target"]:
                    closed = True
                elif pos["hold"] >= params.max_hold_bars:
                    closed = True
                elif is_last_bar_of_day:
                    closed = True
                if closed:
                    open_short = None

            # Confirm or discard pending entries
            if pending_long is not None and open_long is None:
                if close_px > pending_long["signal_px"]:
                    atr_p = pending_long["atr"]
                    open_long = {
                        "entry": close_px,
                        "stop": close_px - atr_p * params.atr_stop_mult,
                        "target": close_px + atr_p * params.atr_target_mult,
                        "entry_time": bar_ts,
                        "hold": 0,
                    }
                pending_long = None

            if pending_short is not None and open_short is None:
                if close_px < pending_short["signal_px"]:
                    atr_p = pending_short["atr"]
                    open_short = {
                        "entry": close_px,
                        "stop": close_px + atr_p * params.atr_stop_mult,
                        "target": close_px - atr_p * params.atr_target_mult,
                        "entry_time": bar_ts,
                        "hold": 0,
                    }
                pending_short = None

            if day_bar_count[bar_date] <= 15 or is_last_bar_of_day:
                continue

            atr = float(indicators["atr"])
            adx = float(indicators["adx"])
            vol_ratio = float(indicators["volume_ratio"])
            rsi = float(indicators["rsi"])
            prev_rsi = float(ind.iloc[i - 1]["rsi"]) if i > 0 else rsi
            bb_lower = float(indicators["bb_lower"])
            bb_upper = float(indicators["bb_upper"])

            if adx > params.adx_max:
                continue

            open_px = float(bar["Open"])
            high_px = float(bar["High"])
            low_px  = float(bar["Low"])

            if params.require_reversal_candle:
                body        = abs(close_px - open_px)
                lower_wick  = min(open_px, close_px) - low_px
                upper_wick  = high_px - max(open_px, close_px)
                wr          = params.candle_wick_ratio
                is_rev_long  = (close_px > open_px) or (body > 0 and lower_wick >= body * wr)
                is_rev_short = (close_px < open_px) or (body > 0 and upper_wick >= body * wr)
            else:
                is_rev_long = is_rev_short = True

            rsi_turn_long  = (not params.rsi_turn) or (rsi > prev_rsi)
            rsi_turn_short = (not params.rsi_turn) or (rsi < prev_rsi)

            rth_open_px = rth_open.get(bar_date)
            from_open_pct = ((close_px - rth_open_px) / rth_open_px * 100.0
                             if rth_open_px else 0.0)
            long_blocked_by_drop = (
                params.from_open_drop_max > 0
                and rth_open_px is not None
                and from_open_pct < -params.from_open_drop_max
            )

            if (params.sides in ("both", "long_only")
                    and open_long is None and pending_long is None
                    and not long_blocked_by_drop
                    and rsi < params.rsi_long_thresh
                    and close_px <= bb_lower
                    and vol_ratio >= params.vol_min
                    and is_rev_long and rsi_turn_long):
                if params.entry_confirm:
                    pending_long = {"signal_px": close_px, "atr": atr}
                else:
                    open_long = {
                        "entry": close_px,
                        "stop": close_px - atr * params.atr_stop_mult,
                        "target": close_px + atr * params.atr_target_mult,
                        "entry_time": bar_ts,
                        "hold": 0,
                    }

            if (params.sides in ("both", "short_only")
                    and open_short is None and pending_short is None
                    and rsi > params.rsi_short_thresh
                    and close_px >= bb_upper
                    and vol_ratio >= params.vol_min
                    and is_rev_short and rsi_turn_short):
                if params.entry_confirm:
                    pending_short = {"signal_px": close_px, "atr": atr}
                else:
                    open_short = {
                        "entry": close_px,
                        "stop": close_px + atr * params.atr_stop_mult,
                        "target": close_px - atr * params.atr_target_mult,
                        "entry_time": bar_ts,
                        "hold": 0,
                    }

        # Translate to open position records (only today's)
        for side, pos in (("long", open_long), ("short", open_short)):
            if pos is not None and _bar_date(pos["entry_time"]) == today:
                self._open[symbol] = {
                    "side": side,
                    "entry_px": pos["entry"],
                    "stop_px": pos["stop"],
                    "target_px": pos["target"],
                    "entry_time": _iso(pos["entry_time"]),
                    "hold_bars": pos["hold"],
                }
                return
        # No open position found for today
        if self._open.get(symbol) is not None:
            if _bar_date_str(self._open[symbol]["entry_time"]) != today:
                self._open[symbol] = None

    def _trade_to_record(self, trade: Any, today: str) -> dict:
        entry = float(trade.entry_price)
        exit_p = float(trade.exit_price)
        if trade.side == "long":
            ret_pct = (exit_p - entry) / entry * 100.0
        else:
            ret_pct = (entry - exit_p) / entry * 100.0
        dollar_pnl = ret_pct / 100.0 * self.slot_size
        return {
            "date": today,
            "symbol": trade.symbol,
            "side": trade.side,
            "entry_time": _iso(trade.entry_time),
            "exit_time": _iso(trade.exit_time),
            "entry_px": round(entry, 4),
            "exit_px": round(exit_p, 4),
            "exit_reason": trade.exit_reason,
            "hold_bars": int(trade.hold_bars),
            "return_pct": round(ret_pct, 4),
            "dollar_pnl": round(dollar_pnl, 2),
            "slot_size": self.slot_size,
            "open": False,
        }

    def _append_ndjson(self, record: dict) -> None:
        NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NDJSON_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def _load_state(self) -> None:
        """Load logged trade keys from existing ndjson to survive restarts."""
        if not NDJSON_PATH.exists():
            return
        today = _today_et()
        try:
            with NDJSON_PATH.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        key = (rec["symbol"], rec["side"], rec["entry_time"])
                        self._logged.add(key)
                        if rec.get("date") == today:
                            self._today_trades.append(rec)
                    except (json.JSONDecodeError, KeyError):
                        pass
            self._today_date = today
        except Exception as exc:
            logger.warning("scalp_shadow state load failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(ts: Any) -> str:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if isinstance(ts, datetime):
        return ts.isoformat()
    return str(ts)


def _bar_date(ts: Any) -> str:
    """Return ET calendar date string for a bar timestamp."""
    from zoneinfo import ZoneInfo
    if isinstance(ts, pd.Timestamp):
        dt = ts.to_pydatetime()
    elif isinstance(ts, datetime):
        dt = ts
    else:
        return str(ts)[:10]
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")


def _bar_date_str(ts_str: str) -> str:
    """Parse a stored ISO string and return ET date."""
    try:
        dt = datetime.fromisoformat(ts_str)
        return _bar_date(dt)
    except ValueError:
        return ts_str[:10]


# ---------------------------------------------------------------------------
# Module-level singleton accessed by api/scalp_shadow_api.py
# ---------------------------------------------------------------------------

_engine: ScalpShadowEngine | None = None


def get_engine() -> ScalpShadowEngine:
    global _engine
    if _engine is None:
        _engine = ScalpShadowEngine()
    return _engine
