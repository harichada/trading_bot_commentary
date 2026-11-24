"""Alpaca paper-trading engine for the scalp reversal strategy.

Same 60-second poll cadence as the shadow loop. Each tick:
  1. Load latest bars per symbol (Postgres + fixed yfinance tail).
  2. For bars newer than last-processed, run incremental signal detection.
  3. On signal: submit a market order via Alpaca paper TradingClient.
  4. For open positions: check stop / target / quick_profit / timeout / EOD.
  5. Close via market order when any exit condition triggers.

Completed trades → data/paper_scalp_reversal.ndjson
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import DEFAULT_DSN, compute_indicators
from backtest.scalp_reversal import ScalpParams, load_bars
from core.scalp_live import LIVE_DEFAULT_PARAMS
from core.scalp_regime import stock_sides
from core.scalp_shadow import (
    DEFAULT_SHADOW_PARAMS,
    DEFAULT_SYMBOLS,
    SLOT_SIZE_USD,
    _bar_date,
    _iso,
    _is_rth,
    _now_et,
    _today_et,
)

logger = logging.getLogger("TradingBot")

_REPO_ROOT = Path(__file__).resolve().parent.parent
NDJSON_PATH = _REPO_ROOT / "data" / "paper_scalp_reversal.ndjson"

_REQUIRED_IND = {
    "rsi", "macd", "macd_signal", "bb_lower", "bb_middle",
    "bb_upper", "atr", "adx", "sma_20", "sma_50",
    "volume_ratio", "high_20", "low_20",
}


class _SymState:
    """Incremental per-symbol position + signal state."""
    __slots__ = (
        "open_long", "open_short",
        "long_cd", "short_cd",
        "day_bar_count", "rth_open",
        "current_date",
    )

    def __init__(self) -> None:
        self.open_long: dict | None = None
        self.open_short: dict | None = None
        self.long_cd: int = 0
        self.short_cd: int = 0
        self.day_bar_count: int = 0
        self.rth_open: float | None = None
        self.current_date: str = ""


class ScalpPaperEngine:
    """Paper-trades the scalp reversal signal via Alpaca paper account."""

    def __init__(
        self,
        symbols: list[str] = DEFAULT_SYMBOLS,
        params: ScalpParams = LIVE_DEFAULT_PARAMS,
        dsn: str = DEFAULT_DSN,
        slot_size: int = SLOT_SIZE_USD,
        tick_interval: int = 60,
        max_slots: int = 3,
    ) -> None:
        self._universe = symbols
        self.symbols   = list(symbols)
        self.params    = params
        self.dsn       = dsn
        self.slot_size = slot_size
        self.tick_interval = tick_interval
        self.max_slots = max_slots

        self._state: dict[str, _SymState] = {s: _SymState() for s in symbols}
        # symbol → {side, entry_px, stop_px, target_px, entry_time, qty, hold}
        self._open: dict[str, dict] = {}
        # symbol → last processed bar timestamp
        self._last_ts: dict[str, Any] = {}
        self._today_trades: list[dict] = []
        self._today_date: str = ""
        self._task: asyncio.Task | None = None
        self._client = None   # TradingClient, created lazily
        # 3 consecutive losses → block symbol for the rest of the day
        self._consec_losses: dict[str, int] = {s: 0 for s in symbols}
        self._blocked_today: set[str] = set()
        self._daily_pnl:       dict[str, float] = {s: 0.0 for s in symbols}
        self._caution_syms:    set[str]         = set()   # tier-1: gate required + half size
        self.loss_caution_usd: float            = 150.0
        self.loss_block_usd:   float            = 300.0
        # Per-symbol direction gate: "long_only" | "short_only" | "both"
        self._symbol_sides: dict[str, str] = {}

    # ── public ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop(), name="scalp_paper_loop")
        logger.info("scalp_paper started symbols=%d interval=%ds",
                    len(self.symbols), self.tick_interval)

    def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def get_status(self) -> dict:
        today = _today_et()
        trades = [t for t in self._today_trades if t["date"] == today]
        wins = [t for t in trades if t["dollar_pnl"] > 0]
        total_pnl = sum(t["dollar_pnl"] for t in trades)
        win_rate = len(wins) / len(trades) * 100 if trades else 0.0
        return {
            "date": today,
            "is_rth": _is_rth(),
            "running": self._task is not None and not self._task.done(),
            "params": asdict(self.params),
            "symbols":        self._universe,
            "active_symbols": self.symbols,
            "blocked_today":             sorted(self._blocked_today),
            "daily_pnl_by_symbol": {s: round(v, 2) for s, v in self._daily_pnl.items() if v != 0.0},
            "caution_symbols":     sorted(self._caution_syms - self._blocked_today),
            "loss_caution_usd":    self.loss_caution_usd,
            "loss_block_usd":      self.loss_block_usd,
            "slot_size": self.slot_size,
            "max_slots": self.max_slots,
            "summary": {
                "n_trades": len(trades),
                "n_open": len(self._open),
                "total_pnl": round(total_pnl, 2),
                "win_rate_pct": round(win_rate, 1),
                "n_wins": len(wins),
                "n_stops": sum(1 for t in trades if t.get("exit_reason") == "stop"),
            },
            "open_positions": [
                {
                    "symbol": sym,
                    **{k: pos[k] for k in
                       ("side", "entry_time", "entry_px", "stop_px", "target_px", "hold", "qty")},
                }
                for sym, pos in self._open.items()
            ],
            "completed_trades": sorted(
                trades, key=lambda t: t.get("exit_time", ""), reverse=True
            ),
        }

    # ── internal loop ─────────────────────────────────────────────────────

    async def _loop(self) -> None:
        self._client = await asyncio.to_thread(self._make_client)
        if self._client is None:
            return
        while True:
            try:
                today = _today_et()
                if today != self._today_date:
                    self._on_new_day(today)
                await asyncio.to_thread(self._tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("scalp_paper tick error: %s", exc, exc_info=True)
            await asyncio.sleep(self.tick_interval)

    def _make_client(self):
        try:
            from alpaca.trading.client import TradingClient
            is_paper = os.getenv("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
            if is_paper:
                key    = os.getenv("ALPACA_PAPER_KEY")    or os.getenv("ALPACA_API_KEY")
                secret = os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET_KEY")
            else:
                key    = os.getenv("ALPACA_API_KEY")
                secret = os.getenv("ALPACA_SECRET_KEY")
            if not key or not secret:
                logger.error("scalp_paper: no Alpaca credentials found in env")
                return None
            client = TradingClient(key, secret, paper=is_paper)
            acct = client.get_account()
            mode = "paper" if is_paper else "LIVE"
            logger.info("scalp_paper: connected to Alpaca %s account "
                        "equity=$%.2f buying_power=$%.2f",
                        mode, float(acct.equity), float(acct.buying_power))
            return client
        except Exception as exc:
            logger.error("scalp_paper: Alpaca connection failed: %s", exc)
            return None

    def _refresh_symbols(self) -> None:
        """Replace active symbol list with today's top movers from screener."""
        try:
            from core.scalp_screener import select_symbols
            scored = select_symbols(top_n=max(len(self._universe), 15))
            if scored:
                for s in scored:
                    if s.symbol not in self._state:
                        self._state[s.symbol] = _SymState()
                self.symbols = [s.symbol for s in scored]
                self._symbol_sides = {s.symbol: stock_sides(s.move_pct) for s in scored}
                logger.info("scalp_paper screener: %d symbols → %s",
                            len(self.symbols),
                            [(s.symbol, self._symbol_sides[s.symbol]) for s in scored[:4]])
                return
        except Exception as exc:
            logger.warning("scalp_paper screener failed, keeping universe: %s", exc)
        self.symbols = list(self._universe)
        self._symbol_sides = {}

    def _on_new_day(self, date: str) -> None:
        self._today_date = date
        self._today_trades = []
        self._last_ts.clear()
        self._refresh_symbols()
        self._consec_losses  = {s: 0 for s in self.symbols}
        self._blocked_today  = set()
        self._daily_pnl      = {s: 0.0 for s in self.symbols}
        self._caution_syms   = set()
        for s in self._state.values():
            s.day_bar_count = 0
            s.rth_open = None
            s.current_date = date
        logger.info("scalp_paper new day %s — %d active symbols", date, len(self.symbols))

    def _record_loss_streak(self, symbol: str, dollar_pnl: float) -> None:
        import time as _time
        self._daily_pnl[symbol] = self._daily_pnl.get(symbol, 0.0) + dollar_pnl
        dpnl = self._daily_pnl[symbol]

        # Tier 2: permanent block
        if dpnl < -self.loss_block_usd and symbol not in self._blocked_today:
            self._blocked_today.add(symbol)
            self._caution_syms.discard(symbol)
            logger.info("scalp_paper %s BLOCKED (daily loss $%.2f > $%.2f)",
                        symbol, dpnl, self.loss_block_usd)
            return

        # Tier 1: direction gate required + half size
        if dpnl < -self.loss_caution_usd and symbol not in self._blocked_today:
            self._caution_syms.add(symbol)
            logger.info("scalp_paper %s caution — daily loss $%.2f, direction gate + 50%% size",
                        symbol, dpnl)

        # Consecutive loss guard (3 in a row → permanent block)
        if dollar_pnl < 0:
            self._consec_losses[symbol] = self._consec_losses.get(symbol, 0) + 1
            if self._consec_losses[symbol] >= 3 and symbol not in self._blocked_today:
                self._blocked_today.add(symbol)
                logger.info("scalp_paper %s blocked (3 consecutive losses)", symbol)
        else:
            self._consec_losses[symbol] = 0

    # ── tick ──────────────────────────────────────────────────────────────

    def _tick(self) -> None:
        today = _today_et()
        for symbol in self.symbols:
            if symbol in self._blocked_today:
                continue
            try:
                df = load_bars(symbol, days=5, dsn=self.dsn)
                if df.empty or len(df) < 51:
                    continue
                if _bar_date(df.index[-1]) != today:
                    continue
                self._process_symbol(symbol, df, today)
            except Exception as exc:
                logger.debug("scalp_paper %s tick error: %s", symbol, exc)

    def _process_symbol(self, symbol: str, df: pd.DataFrame, today: str) -> None:
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")

        last = self._last_ts.get(symbol)
        if last is not None:
            new_df = df[df.index > last]
        else:
            # First tick: seed state from recent history, only act on latest bar
            new_df = df.iloc[-1:]

        if new_df.empty:
            return

        ind = compute_indicators(df)
        state = self._state[symbol]

        for idx in range(len(new_df)):
            bar_ts    = new_df.index[idx]
            bar_pos   = df.index.get_loc(bar_ts)
            bar       = df.iloc[bar_pos]
            ind_row   = ind.iloc[bar_pos]

            # Update day counter
            bar_et    = bar_ts.tz_convert(ET)
            bar_date  = bar_et.date().isoformat()
            if bar_date != state.current_date:
                state.current_date  = bar_date
                state.day_bar_count = 0
                state.rth_open      = float(bar["Open"])
            state.day_bar_count += 1

            self._process_bar(symbol, bar, bar_ts, bar_et, ind_row, ind, bar_pos, state, today, hist_df=df)

        self._last_ts[symbol] = new_df.index[-1]

    def _process_bar(
        self,
        symbol: str,
        bar: pd.Series,
        bar_ts: Any,
        bar_et: Any,
        ind_row: pd.Series,
        ind: pd.DataFrame,
        bar_pos: int,
        state: _SymState,
        today: str,
        hist_df: "pd.DataFrame | None" = None,
    ) -> None:
        if any(pd.isna(ind_row.get(k)) for k in _REQUIRED_IND):
            return

        params    = self.params
        close_px  = float(bar["Close"])
        high_px   = float(bar["High"])
        low_px    = float(bar["Low"])
        bar_min   = bar_et.hour * 60 + bar_et.minute
        is_eod    = bar_min >= 15 * 60 + 55

        # ── manage open long ──────────────────────────────────────────────
        pos = self._open.get(symbol)
        if pos and pos["side"] == "long":
            pos["hold"] += 1
            entry = pos["entry_px"]
            unr   = (close_px - entry) / entry * 100.0
            reason = exit_px = None
            if low_px <= pos["stop_px"]:
                reason, exit_px = "stop",         pos["stop_px"]
            elif high_px >= pos["target_px"]:
                reason, exit_px = "target",        pos["target_px"]
            elif pos["hold"] <= params.quick_exit_bars and unr >= params.min_profit_pct:
                reason, exit_px = "quick_profit",  close_px
            elif pos["hold"] >= params.max_hold_bars:
                reason, exit_px = "timeout",       close_px
            elif is_eod:
                reason, exit_px = "eod",           close_px
            if reason:
                self._close(symbol, pos, reason, exit_px, bar_ts, today)
                return

        # ── manage open short ─────────────────────────────────────────────
        elif pos and pos["side"] == "short":
            pos["hold"] += 1
            entry = pos["entry_px"]
            unr   = (entry - close_px) / entry * 100.0
            reason = exit_px = None
            if high_px >= pos["stop_px"]:
                reason, exit_px = "stop",         pos["stop_px"]
            elif low_px <= pos["target_px"]:
                reason, exit_px = "target",        pos["target_px"]
            elif pos["hold"] <= params.quick_exit_bars and unr >= params.min_profit_pct:
                reason, exit_px = "quick_profit",  close_px
            elif pos["hold"] >= params.max_hold_bars:
                reason, exit_px = "timeout",       close_px
            elif is_eod:
                reason, exit_px = "eod",           close_px
            if reason:
                self._close(symbol, pos, reason, exit_px, bar_ts, today)
                return

        # ── skip signal: warm-up / EOD / position already open ────────────
        if state.day_bar_count <= 15 or is_eod or symbol in self._open:
            return
        if len(self._open) >= self.max_slots:
            return

        # ── cooldown ──────────────────────────────────────────────────────
        if state.long_cd > 0:
            state.long_cd -= 1
        if state.short_cd > 0:
            state.short_cd -= 1

        # ── indicators ────────────────────────────────────────────────────
        atr       = float(ind_row["atr"])
        adx       = float(ind_row["adx"])
        vol_ratio = float(ind_row["volume_ratio"])
        rsi       = float(ind_row["rsi"])
        prev_rsi  = float(ind.iloc[bar_pos - 1]["rsi"]) if bar_pos > 0 else rsi
        bb_lower  = float(ind_row["bb_lower"])
        bb_upper  = float(ind_row["bb_upper"])

        if adx > params.adx_max:
            return

        rth_open_px = state.rth_open
        from_open   = ((close_px - rth_open_px) / rth_open_px * 100.0
                       if rth_open_px else 0.0)
        long_blocked = (
            params.from_open_drop_max > 0
            and rth_open_px is not None
            and from_open < -params.from_open_drop_max
        )

        rsi_turn_long  = (not params.rsi_turn) or (rsi > prev_rsi)
        rsi_turn_short = (not params.rsi_turn) or (rsi < prev_rsi)
        is_rev_long = is_rev_short = True

        if params.require_reversal_candle:
            open_px    = float(bar["Open"])
            body       = abs(close_px - open_px)
            lower_wick = min(open_px, close_px) - low_px
            upper_wick = high_px - max(open_px, close_px)
            wr         = params.candle_wick_ratio
            is_rev_long  = (close_px > open_px) or (body > 0 and lower_wick >= body * wr)
            is_rev_short = (close_px < open_px) or (body > 0 and upper_wick >= body * wr)

        bb_long_cond  = close_px <= bb_lower
        bb_short_cond = close_px >= bb_upper

        # Per-stock direction gate: screener tags each symbol long_only or short_only.
        _sym_sides   = self._symbol_sides.get(symbol, "both")
        _allow_long  = params.sides in ("both", "long_only")  and _sym_sides in ("both", "long_only")
        _allow_short = params.sides in ("both", "short_only") and _sym_sides in ("both", "short_only")

        # ── long signal ───────────────────────────────────────────────────
        if (_allow_long
                and state.long_cd == 0
                and not long_blocked
                and rsi < params.rsi_long_thresh
                and bb_long_cond
                and vol_ratio >= params.vol_min
                and is_rev_long and rsi_turn_long):
            eff_slot = self._gate_slot(symbol, "long", hist_df, ind)
            if eff_slot > 0:
                self._enter(symbol, "long", close_px, atr, bar_ts, today, eff_slot)

        # ── short signal ──────────────────────────────────────────────────
        elif (_allow_short
                and state.short_cd == 0
                and rsi > params.rsi_short_thresh
                and bb_short_cond
                and vol_ratio >= params.vol_min
                and is_rev_short and rsi_turn_short):
            eff_slot = self._gate_slot(symbol, "short", hist_df, ind)
            if eff_slot > 0:
                self._enter(symbol, "short", close_px, atr, bar_ts, today, eff_slot)

    def _gate_slot(self, symbol: str, side: str,
                   hist_df: "pd.DataFrame | None", ind: "pd.DataFrame") -> int:
        """Return effective slot size; 0 means direction gate rejected entry."""
        if symbol not in self._caution_syms:
            return self.slot_size
        from core.scalp_direction_gate import direction_gate
        h = hist_df if hist_df is not None else pd.DataFrame()
        allowed, reason = direction_gate(
            symbol, side, h, ind, data_client=self._get_data_client()
        )
        logger.info("scalp_paper direction_gate %s %s: %s", symbol, side, reason)
        if not allowed:
            return 0
        return max(1, int(self.slot_size * 0.5))

    def _get_data_client(self):
        """Lazy-init Alpaca StockHistoricalDataClient for L2/news gate signals."""
        if hasattr(self, "_data_client_cached"):
            return self._data_client_cached
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            is_paper = os.getenv("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
            key    = (os.getenv("ALPACA_PAPER_KEY") or os.getenv("ALPACA_API_KEY") if is_paper
                      else os.getenv("ALPACA_API_KEY"))
            secret = (os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET_KEY") if is_paper
                      else os.getenv("ALPACA_SECRET_KEY"))
            self._data_client_cached = StockHistoricalDataClient(key, secret)
        except Exception:
            self._data_client_cached = None
        return self._data_client_cached

    # ── order submission ──────────────────────────────────────────────────

    def _enter(
        self,
        symbol: str,
        side: str,
        entry_px: float,
        atr: float,
        bar_ts: Any,
        today: str,
        slot_size: int | None = None,
    ) -> None:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        effective_slot = slot_size if slot_size is not None else self.slot_size
        shares    = max(1, int(effective_slot / entry_px))
        stop_px   = (entry_px - atr * self.params.atr_stop_mult   if side == "long"
                     else entry_px + atr * self.params.atr_stop_mult)
        target_px = (entry_px + atr * self.params.atr_target_mult if side == "long"
                     else entry_px - atr * self.params.atr_target_mult)
        try:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY if side == "long" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self._client.submit_order(req)
            order_id = str(order.id)
        except Exception as exc:
            logger.error("scalp_paper ENTRY FAILED %s %s: %s", symbol, side, exc)
            return

        self._open[symbol] = {
            "side":       side,
            "entry_px":   round(entry_px, 4),
            "stop_px":    round(stop_px, 4),
            "target_px":  round(target_px, 4),
            "entry_time": _iso(bar_ts),
            "qty":        shares,
            "hold":       0,
            "order_id":   order_id,
            "slot_size":  effective_slot,
        }
        caution = " [CAUTION 50%]" if effective_slot < self.slot_size else ""
        logger.info(
            "scalp_paper ENTER %s %s qty=%d entry=%.4f stop=%.4f target=%.4f%s",
            symbol, side, shares, entry_px, stop_px, target_px, caution,
        )

    def _close(
        self,
        symbol: str,
        pos: dict,
        reason: str,
        exit_px: float,
        bar_ts: Any,
        today: str,
    ) -> None:
        try:
            self._client.close_position(symbol)
        except Exception as exc:
            logger.error("scalp_paper EXIT FAILED %s: %s", symbol, exc)

        entry     = pos["entry_px"]
        pos_slot  = pos.get("slot_size", self.slot_size)
        ret       = ((exit_px - entry) / entry if pos["side"] == "long"
                     else (entry - exit_px) / entry) * 100.0
        dollar    = ret / 100.0 * pos_slot
        record = {
            "date":        today,
            "symbol":      symbol,
            "side":        pos["side"],
            "entry_time":  pos["entry_time"],
            "exit_time":   _iso(bar_ts),
            "entry_px":    round(entry, 4),
            "exit_px":     round(exit_px, 4),
            "exit_reason": reason,
            "hold_bars":   pos["hold"],
            "return_pct":  round(ret, 4),
            "dollar_pnl":  round(dollar, 2),
            "qty":         pos["qty"],
            "slot_size":   self.slot_size,
        }
        NDJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with NDJSON_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self._today_trades.append(record)
        self._record_loss_streak(symbol, dollar)
        del self._open[symbol]

        if reason == "stop":
            self._state[symbol].long_cd  = self.params.cooldown_bars
            self._state[symbol].short_cd = self.params.cooldown_bars

        logger.info(
            "scalp_paper EXIT %s %s reason=%s exit=%.4f pnl=$%.2f",
            symbol, pos["side"], reason, exit_px, dollar,
        )


# ── module-level singleton ────────────────────────────────────────────────

_engine: ScalpPaperEngine | None = None


def get_engine() -> ScalpPaperEngine:
    global _engine
    if _engine is None:
        _engine = ScalpPaperEngine()
    return _engine
