"""Event-driven scalp reversal engine — Alpaca data, no Schwab stream dependency.

Data flow:
  1. Startup: seed 5-day 1-min history from Alpaca for warm-up.
  2. Runtime: background thread polls Alpaca for the latest closed 1-min bar
     every ~15s per symbol, fires on_bar() when a new bar key is detected.
     Also hooks into SchwabBarCache if the stream is up (fires first if available).
  3. on_bar(): append → compute indicators → check signal → market order.

Completed trades → data/live_scalp_reversal.ndjson
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import compute_indicators
from backtest.scalp_reversal import ScalpParams
from core.scalp_regime import stock_sides
from core.scalp_shadow import (
    DEFAULT_SHADOW_PARAMS,
    DEFAULT_SYMBOLS,
    SLOT_SIZE_USD,
    _is_rth,
    _today_et,
)

# Live engine allows both directions; per-stock daily direction gates which
# side is active per symbol.  Shadow keeps long_only as its validated baseline.
from dataclasses import replace as _dc_replace
LIVE_DEFAULT_PARAMS: ScalpParams = _dc_replace(DEFAULT_SHADOW_PARAMS, sides="both")

logger = logging.getLogger("TradingBot")

_REPO_ROOT  = Path(__file__).resolve().parent.parent
NDJSON_PATH = _REPO_ROOT / "data" / "live_scalp_reversal.ndjson"
_ET         = __import__("zoneinfo").ZoneInfo("America/New_York")

_REQUIRED_IND = {
    "rsi", "macd", "macd_signal", "bb_lower", "bb_upper", "bb_middle",
    "atr", "adx", "sma_20", "sma_50", "volume_ratio", "high_20", "low_20",
}
_HISTORY_BARS = 300
_WARMUP_BARS  = 15


def _iso(ts: Any) -> str:
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _seed_from_alpaca(symbols: list[str], key: str, secret: str) -> dict[str, pd.DataFrame]:
    """Fetch 5 days of 1-min RTH bars per symbol from Alpaca."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(key, secret)
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(days=7)   # extra buffer for weekends
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed="iex",
    )
    try:
        bars = client.get_stock_bars(req)
        raw  = bars.df  # MultiIndex (symbol, timestamp)
    except Exception as exc:
        logger.error("scalp_live seed failed: %s", exc)
        return {}

    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = raw.loc[sym].copy() if sym in raw.index.get_level_values(0) else pd.DataFrame()
            if df.empty:
                continue
            # Rename to title-case expected by compute_indicators
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                     "close": "Close", "volume": "Volume"})
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            # RTH filter
            et_idx  = df.index.tz_convert(_ET)
            minutes = et_idx.hour * 60 + et_idx.minute
            df = df[(minutes >= 9 * 60 + 30) & (minutes <= 15 * 60 + 55)]
            df = df[~df.index.duplicated(keep="last")]
            result[sym] = df.iloc[-_HISTORY_BARS:]
            logger.debug("scalp_live seeded %s: %d bars", sym, len(df))
        except Exception as exc:
            logger.warning("scalp_live seed %s: %s", sym, exc)
    return result


class ScalpLiveEngine:
    """Processes each 1-min bar immediately as it arrives from Schwab stream."""

    def __init__(
        self,
        symbols: list[str] = DEFAULT_SYMBOLS,
        params: ScalpParams = LIVE_DEFAULT_PARAMS,
        slot_size: int = SLOT_SIZE_USD,
        max_slots: int = 2,
    ) -> None:
        self._universe = symbols          # static fallback universe
        self.symbols   = list(symbols)   # active symbol list (refreshed daily)
        self.params    = params
        self.slot_size = slot_size
        self.max_slots = max_slots

        self._history:      dict[str, pd.DataFrame] = {}
        self._bar_count:    dict[str, int]           = {s: 0 for s in symbols}
        self._open:         dict[str, dict]          = {}
        self._pending:      set[str]                 = set()
        self._long_cd:      dict[str, int]           = {s: 0 for s in symbols}
        self._short_cd:     dict[str, int]           = {s: 0 for s in symbols}
        self._consec:           dict[str, int]           = {s: 0 for s in symbols}
        self._blocked:          set[str]                 = set()
        self._daily_pnl:        dict[str, float]         = {s: 0.0 for s in symbols}
        self._caution_syms:     set[str]                 = set()  # tier-1: gate required
        self.loss_caution_usd:  float                    = 150.0  # tier-1 threshold
        self.loss_block_usd:    float                    = 300.0  # tier-2: permanent block
        self._today_date:   str                      = ""
        self._today_trades: list[dict]               = []
        self._client       = None   # Alpaca TradingClient
        self._running      = False
        self._lock         = threading.Lock()
        self._seen_bars:    dict[str, int]           = {}   # symbol → last ts_ms seen
        # Per-symbol direction gate: "long_only" | "short_only" | "both"
        # Set by screener (gainer → short_only, loser → long_only).
        self._symbol_sides: dict[str, str] = {}

    # ── public ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._client = self._make_trading_client()
        if self._client is None:
            return

        is_paper = os.getenv("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
        key    = (os.getenv("ALPACA_PAPER_KEY") or os.getenv("ALPACA_API_KEY")) if is_paper \
                 else os.getenv("ALPACA_API_KEY")
        secret = (os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET_KEY")) if is_paper \
                 else os.getenv("ALPACA_SECRET_KEY")

        logger.info("scalp_live: seeding %d symbols from Alpaca…", len(self.symbols))
        self._history = _seed_from_alpaca(self.symbols, key, secret)
        logger.info("scalp_live: seeded %d/%d symbols", len(self._history), len(self.symbols))

        # Register with Schwab stream if available (fires first when healthy)
        try:
            from core.schwab_bar_cache import SchwabBarCache
            cache = SchwabBarCache.instance()
            cache.register_bar_callback(self._on_bar)
            cache.request_chart_symbols(set(self.symbols))
            logger.info("scalp_live: stream callback registered, requested CHART_EQUITY for %d symbols",
                        len(self.symbols))
        except Exception as exc:
            logger.warning("scalp_live: stream callback failed: %s", exc)

        # Always start the Alpaca poll loop as primary/fallback bar source
        self._running = True
        self._seen_bars: dict[str, int] = {}   # symbol → last ts_ms seen
        threading.Thread(target=self._alpaca_poll_loop, daemon=True).start()
        logger.info("scalp_live: started — Alpaca poll loop running")

    def stop(self) -> None:
        self._running = False

    def get_status(self) -> dict:
        today  = _today_et()
        trades = [t for t in self._today_trades if t["date"] == today]
        wins   = [t for t in trades if t["dollar_pnl"] > 0]
        total  = sum(t["dollar_pnl"] for t in trades)
        return {
            "date":      today,
            "is_rth":    _is_rth(),
            "running":   self._running,
            "params":    asdict(self.params),
            "symbols":         self._universe,
            "active_symbols":  self.symbols,
            "blocked":                   sorted(self._blocked),
            "daily_pnl_by_symbol":  {s: round(v, 2) for s, v in self._daily_pnl.items() if v != 0.0},
            "caution_symbols":      sorted(self._caution_syms - self._blocked),
            "loss_caution_usd":     self.loss_caution_usd,
            "loss_block_usd":       self.loss_block_usd,
            "slot_size": self.slot_size,
            "max_slots": self.max_slots,
            "summary": {
                "n_trades":     len(trades),
                "n_open":       len(self._open),
                "total_pnl":    round(total, 2),
                "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else 0.0,
                "n_wins":       len(wins),
            },
            "open_positions": list(self._open.values()),
            "completed_trades": sorted(trades, key=lambda t: t.get("exit_time", ""), reverse=True),
        }

    # ── bar callback ──────────────────────────────────────────────────────

    def _on_bar(self, symbol: str, ts_ms: int,
                open_: float, high: float, low: float,
                close: float, volume: float) -> None:
        if not self._running or symbol not in self.symbols:
            return
        # Cross-source dedup: Schwab stream and Alpaca poll both feed this engine.
        # Mark ts_ms as seen here so the poll loop skips it (GIL-safe dict write).
        if self._seen_bars.get(symbol) == ts_ms:
            return
        self._seen_bars[symbol] = ts_ms
        threading.Thread(
            target=self._process_bar,
            args=(symbol, ts_ms, open_, high, low, close, volume),
            daemon=True,
        ).start()

    def _process_bar(self, symbol: str, ts_ms: int,
                     open_: float, high: float, low: float,
                     close: float, volume: float) -> None:
        with self._lock:
            today = _today_et()
            if today != self._today_date:
                self._on_new_day(today)

            if symbol in self._blocked:
                return

            ts_utc  = pd.Timestamp(ts_ms / 1000.0, unit="s", tz="UTC")
            bar_et  = ts_utc.tz_convert(_ET)
            bar_min = bar_et.hour * 60 + bar_et.minute

            if bar_min < 9 * 60 + 30 or bar_min > 15 * 60 + 55:
                return

            new_row = pd.DataFrame(
                [{"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}],
                index=pd.DatetimeIndex([ts_utc]),
            )
            hist = self._history.get(symbol, pd.DataFrame())
            if not hist.empty and ts_utc in hist.index:
                return
            hist = pd.concat([hist, new_row]).iloc[-_HISTORY_BARS:]
            self._history[symbol] = hist

            if len(hist) < 55:
                return

            self._bar_count[symbol] = self._bar_count.get(symbol, 0) + 1
            if self._bar_count[symbol] <= _WARMUP_BARS:
                return

            try:
                ind = compute_indicators(hist)
            except Exception:
                return
            ind_row = ind.iloc[-1]
            if any(pd.isna(ind_row.get(k)) for k in _REQUIRED_IND):
                return

            is_eod = bar_min >= 15 * 60 + 55

            # ── manage open position ──────────────────────────────────────
            pos = self._open.get(symbol)
            if pos:
                pos["hold"] += 1
                entry  = pos["entry_px"]
                side   = pos["side"]
                reason = exit_px = None
                if side == "long":
                    unr = (close - entry) / entry * 100.0
                    if low <= pos["stop_px"]:
                        reason, exit_px = "stop",         pos["stop_px"]
                    elif high >= pos["target_px"]:
                        reason, exit_px = "target",        pos["target_px"]
                    elif pos["hold"] <= self.params.quick_exit_bars and unr >= self.params.min_profit_pct:
                        reason, exit_px = "quick_profit",  close
                    elif pos["hold"] >= self.params.max_hold_bars:
                        reason, exit_px = "timeout",       close
                    elif is_eod:
                        reason, exit_px = "eod",           close
                else:
                    unr = (entry - close) / entry * 100.0
                    if high >= pos["stop_px"]:
                        reason, exit_px = "stop",         pos["stop_px"]
                    elif low <= pos["target_px"]:
                        reason, exit_px = "target",        pos["target_px"]
                    elif pos["hold"] <= self.params.quick_exit_bars and unr >= self.params.min_profit_pct:
                        reason, exit_px = "quick_profit",  close
                    elif pos["hold"] >= self.params.max_hold_bars:
                        reason, exit_px = "timeout",       close
                    elif is_eod:
                        reason, exit_px = "eod",           close
                if reason:
                    self._close(symbol, pos, reason, exit_px, ts_utc, today)
                return

            # ── entry signal ──────────────────────────────────────────────
            if is_eod or symbol in self._pending or symbol in self._open:
                return
            if len(self._open) + len(self._pending) >= self.max_slots:
                return

            atr       = float(ind_row["atr"])
            adx       = float(ind_row["adx"])
            vol_ratio = float(ind_row["volume_ratio"])
            rsi       = float(ind_row["rsi"])
            prev_rsi  = float(ind.iloc[-2]["rsi"]) if len(ind) > 1 else rsi
            bb_lower  = float(ind_row["bb_lower"])
            bb_upper  = float(ind_row["bb_upper"])

            if adx > self.params.adx_max:
                return
            if self._long_cd.get(symbol, 0) > 0:
                self._long_cd[symbol] -= 1
            if self._short_cd.get(symbol, 0) > 0:
                self._short_cd[symbol] -= 1

            rsi_turn_long  = (not self.params.rsi_turn) or (rsi > prev_rsi)
            rsi_turn_short = (not self.params.rsi_turn) or (rsi < prev_rsi)
            is_rev_long = is_rev_short = True
            if self.params.require_reversal_candle:
                body       = abs(close - open_)
                lower_wick = min(open_, close) - low
                upper_wick = high - max(open_, close)
                wr = self.params.candle_wick_ratio
                is_rev_long  = (close > open_) or (body > 0 and lower_wick >= body * wr)
                is_rev_short = (close < open_) or (body > 0 and upper_wick >= body * wr)

            p = self.params

            # Per-stock direction gate: screener tags each symbol as long_only or
            # short_only based on its daily move (gainer → short, loser → long).
            _sym_sides   = self._symbol_sides.get(symbol, "both")
            _allow_long  = p.sides in ("both", "long_only")  and _sym_sides in ("both", "long_only")
            _allow_short = p.sides in ("both", "short_only") and _sym_sides in ("both", "short_only")

            if (_allow_long
                    and self._long_cd.get(symbol, 0) == 0
                    and rsi < p.rsi_long_thresh
                    and close <= bb_lower
                    and vol_ratio >= p.vol_min
                    and is_rev_long and rsi_turn_long):
                eff_slot = self._symbol_slot_size(symbol, "long", hist, ind)
                if eff_slot > 0:
                    self._enter(symbol, "long", close, atr, ts_utc, today, eff_slot)

            elif (_allow_short
                    and self._short_cd.get(symbol, 0) == 0
                    and rsi > p.rsi_short_thresh
                    and close >= bb_upper
                    and vol_ratio >= p.vol_min
                    and is_rev_short and rsi_turn_short):
                eff_slot = self._symbol_slot_size(symbol, "short", hist, ind)
                if eff_slot > 0:
                    self._enter(symbol, "short", close, atr, ts_utc, today, eff_slot)

    # ── order execution ───────────────────────────────────────────────────

    def _enter(self, symbol: str, side: str, entry_px: float,
               atr: float, bar_ts: Any, today: str,
               slot_size: int | None = None) -> None:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        effective_slot = slot_size if slot_size is not None else self.slot_size
        shares    = max(1, int(effective_slot / entry_px))
        stop_px   = (entry_px - atr * self.params.atr_stop_mult  if side == "long"
                     else entry_px + atr * self.params.atr_stop_mult)
        target_px = (entry_px + atr * self.params.atr_target_mult if side == "long"
                     else entry_px - atr * self.params.atr_target_mult)
        self._pending.add(symbol)
        try:
            order = self._client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=shares,
                side=OrderSide.BUY if side == "long" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            ))
        except Exception as exc:
            logger.error("scalp_live ENTRY FAILED %s: %s", symbol, exc)
            self._pending.discard(symbol)
            return

        self._pending.discard(symbol)
        self._open[symbol] = {
            "symbol":     symbol,
            "side":       side,
            "entry_px":   round(entry_px, 4),
            "stop_px":    round(stop_px, 4),
            "target_px":  round(target_px, 4),
            "entry_time": _iso(bar_ts),
            "qty":        shares,
            "hold":       0,
            "order_id":   str(order.id),
            "slot_size":  effective_slot,
        }
        caution = " [CAUTION 50%]" if effective_slot < self.slot_size else ""
        logger.info("scalp_live ENTER %s %s qty=%d @%.4f stop=%.4f tgt=%.4f%s",
                    symbol, side, shares, entry_px, stop_px, target_px, caution)

    def _close(self, symbol: str, pos: dict, reason: str,
               exit_px: float, bar_ts: Any, today: str) -> None:
        try:
            self._client.close_position(symbol)
        except Exception as exc:
            logger.error("scalp_live EXIT FAILED %s: %s", symbol, exc)

        entry        = pos["entry_px"]
        pos_slot     = pos.get("slot_size", self.slot_size)
        ret          = ((exit_px - entry) / entry if pos["side"] == "long"
                        else (entry - exit_px) / entry) * 100.0
        dollar       = ret / 100.0 * pos_slot
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
            self._long_cd[symbol] = self._short_cd[symbol] = self.params.cooldown_bars
        logger.info("scalp_live EXIT %s %s reason=%s @%.4f pnl=$%.2f",
                    symbol, pos["side"], reason, exit_px, dollar)

    # ── helpers ───────────────────────────────────────────────────────────

    def _alpaca_poll_loop(self) -> None:
        """Poll Alpaca for the latest closed 1-min bar for each symbol.

        Fires on_bar() when a new bar timestamp is detected. Runs every
        ~15s cycling through all symbols (~1s per symbol).
        """
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        is_paper = os.getenv("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
        key    = (os.getenv("ALPACA_PAPER_KEY") or os.getenv("ALPACA_API_KEY")) if is_paper \
                 else os.getenv("ALPACA_API_KEY")
        secret = (os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET_KEY")) if is_paper \
                 else os.getenv("ALPACA_SECRET_KEY")
        data_client = StockHistoricalDataClient(key, secret)

        import time
        while self._running:
            if not _is_rth():
                time.sleep(15)
                continue
            try:
                end   = datetime.now(timezone.utc)
                start = end - timedelta(minutes=5)
                req = StockBarsRequest(
                    symbol_or_symbols=self.symbols,
                    timeframe=TimeFrame.Minute,
                    start=start,
                    end=end,
                    feed="iex",
                )
                bars = data_client.get_stock_bars(req)
                raw  = bars.df  # MultiIndex (symbol, ts)
                lvl0 = raw.index.get_level_values(0)

                for sym in self.symbols:
                    if sym not in lvl0:
                        continue
                    sym_bars = raw.loc[sym]
                    if sym_bars.empty:
                        continue
                    latest = sym_bars.iloc[-1]
                    ts_utc  = latest.name
                    ts_ms   = int(ts_utc.timestamp() * 1000)
                    if self._seen_bars.get(sym) == ts_ms:
                        continue   # already processed this bar
                    self._seen_bars[sym] = ts_ms
                    self._on_bar(
                        sym, ts_ms,
                        float(latest["open"]),
                        float(latest["high"]),
                        float(latest["low"]),
                        float(latest["close"]),
                        float(latest["volume"]),
                    )
            except Exception as exc:
                logger.debug("scalp_live poll error: %s", exc)
            time.sleep(15)

    def _refresh_symbols(self) -> None:
        """Replace active symbol list with today's top movers from screener."""
        from core.scalp_shadow import EXCLUDED_SYMBOLS
        try:
            from core.scalp_screener import select_symbols
            from core.scalp_regime import stock_sides
            scored = [s for s in select_symbols(top_n=max(len(self._universe), 15))
                      if s.symbol not in EXCLUDED_SYMBOLS]
            if scored:
                self.symbols = [s.symbol for s in scored]
                self._symbol_sides = {s.symbol: stock_sides(s.move_pct) for s in scored}
                logger.info("scalp_live screener: %d symbols → %s",
                            len(self.symbols),
                            [(s.symbol, self._symbol_sides[s.symbol]) for s in scored[:4]])
                return
        except Exception as exc:
            logger.warning("scalp_live screener failed, keeping universe: %s", exc)
        self.symbols = [s for s in self._universe if s not in EXCLUDED_SYMBOLS]
        self._symbol_sides = {}   # no direction gate for static universe

    def _on_new_day(self, date: str) -> None:
        self._today_date    = date
        self._today_trades  = []
        self._pending.clear()
        self._refresh_symbols()

        # Update Schwab CHART_EQUITY subscriptions for today's symbol list
        try:
            from core.schwab_bar_cache import SchwabBarCache
            SchwabBarCache.instance().request_chart_symbols(set(self.symbols))
            logger.info("scalp_live: updated CHART_EQUITY request for %d symbols", len(self.symbols))
        except Exception:
            pass
        self._bar_count     = {s: 0 for s in self.symbols}
        self._consec        = {s: 0 for s in self.symbols}
        self._daily_pnl     = {s: 0.0 for s in self.symbols}
        self._blocked       = set()
        self._caution_syms  = set()
        self._long_cd       = {s: 0 for s in self.symbols}
        self._short_cd      = {s: 0 for s in self.symbols}
        logger.info("scalp_live new day %s — %d active symbols", date, len(self.symbols))

    def _record_loss_streak(self, symbol: str, pnl: float) -> None:
        import time as _time
        self._daily_pnl[symbol] = self._daily_pnl.get(symbol, 0.0) + pnl
        dpnl = self._daily_pnl[symbol]

        # Tier 2: permanent block
        if dpnl < -self.loss_block_usd and symbol not in self._blocked:
            self._blocked.add(symbol)
            self._caution_syms.discard(symbol)
            logger.info("scalp_live %s BLOCKED (daily loss $%.2f > $%.2f)",
                        symbol, dpnl, self.loss_block_usd)
            return

        # Tier 1: direction gate required + half size (only escalate)
        if dpnl < -self.loss_caution_usd and symbol not in self._blocked:
            self._caution_syms.add(symbol)
            logger.info("scalp_live %s caution — daily loss $%.2f, direction gate + 50%% size",
                        symbol, dpnl)

        # Consecutive loss guard (3 in a row → permanent block)
        if pnl < 0:
            self._consec[symbol] = self._consec.get(symbol, 0) + 1
            if self._consec[symbol] >= 3 and symbol not in self._blocked:
                self._blocked.add(symbol)
                logger.info("scalp_live %s blocked (3 consecutive losses)", symbol)
        else:
            self._consec[symbol] = 0

    def _symbol_slot_size(self, symbol: str, side: str,
                          history: "pd.DataFrame", ind: "pd.DataFrame") -> int:
        """Return effective slot size; 0 means skip entry (direction gate failed)."""
        if symbol not in self._caution_syms:
            return self.slot_size
        from core.scalp_direction_gate import direction_gate
        allowed, reason = direction_gate(
            symbol, side, history, ind, data_client=self._data_client
        )
        logger.info("scalp_live direction_gate %s %s: %s", symbol, side, reason)
        if not allowed:
            return 0
        return max(1, int(self.slot_size * 0.5))

    def _make_trading_client(self):
        try:
            from alpaca.trading.client import TradingClient
            is_paper = os.getenv("ALPACA_PAPER", "true").lower() not in ("false", "0", "no")
            key    = (os.getenv("ALPACA_PAPER_KEY") or os.getenv("ALPACA_API_KEY")) if is_paper \
                     else os.getenv("ALPACA_API_KEY")
            secret = (os.getenv("ALPACA_PAPER_SECRET") or os.getenv("ALPACA_SECRET_KEY")) if is_paper \
                     else os.getenv("ALPACA_SECRET_KEY")
            if not key or not secret:
                logger.error("scalp_live: no Alpaca credentials")
                return None
            client = TradingClient(key, secret, paper=is_paper)
            acct   = client.get_account()
            mode   = "paper" if is_paper else "LIVE"
            logger.info("scalp_live: Alpaca %s equity=$%.2f", mode, float(acct.equity))
            return client
        except Exception as exc:
            logger.error("scalp_live: Alpaca connect failed: %s", exc)
            return None


# ── singleton ─────────────────────────────────────────────────────────────

_engine: ScalpLiveEngine | None = None


def get_engine() -> ScalpLiveEngine:
    global _engine
    if _engine is None:
        _engine = ScalpLiveEngine()
    return _engine
