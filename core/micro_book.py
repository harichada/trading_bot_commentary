"""v-micro-book-2026-10-05: Level 1 + tape microstructure book.

Spec: docs/superpowers/specs/2026-10-05-adaptive-rsi-turn-design.md §2.

The Schwab stream delivers last, bid, ask, bid_size, ask_size, last_size and
total_volume for every subscribed name; the price book keeps only prices.
This module keeps the rest, in fixed-size rings, O(1) per update:

* trade detection from the volume delta (quote-only updates carry no trade),
* aggressor tagging by the QUOTE rule (live) — the tick rule exists only for
  the historical archive, which has no quotes,
* a 61-slot minute-bin ring with running 15- and 60-minute totals
  (cumulative volume delta, sell share, VWAP, large prints),
* a 900-slot quote ring with EMA bid/ask-size imbalance and EMA spread,
* array-copy snapshots for readers, and a per-session parquet flush.

Nothing here touches pandas on the hot path. One `SymbolMicro` is ~50 KB.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("TradingBot")

BINS = 61                      # 60 minutes plus one slot so the ring never resizes
QUOTE_RING_CAPACITY = 900      # ~15 min at the observed ~1 update/s per name
IMB_HALF_LIFE_S = 60.0
SPR_HALF_LIFE_S = 300.0
LARGE_PRINT_PCTL = 95.0
LARGE_PRINT_SAMPLE = 2048      # print sizes kept for the session percentile

# bin field indices
_BUY, _SELL, _NBUY, _NSELL, _VWAPN, _VOL, _HIGH, _LOW, _CLOSE, _LARGE, _NTR = range(11)
_NFIELDS = 11


def aggressor_tick_rule(price: float, price_prev: Optional[float], side_prev: int) -> int:
    """Historical archive (trades only): +1 on an uptick, -1 on a downtick,
    flat inherits the previous side (0 when nothing is known yet)."""
    if price_prev is None:
        return 0
    if price > price_prev:
        return 1
    if price < price_prev:
        return -1
    return side_prev


def aggressor_quote_rule(last: float, bid_prev: Optional[float], ask_prev: Optional[float],
                         last_prev: Optional[float], side_prev: int) -> int:
    """Live stream: a print at or above the ask that was in force is
    buyer-initiated, at or below the bid seller-initiated; inside the spread
    the tick direction decides and a flat print inherits. Without a quote the
    tick rule is the only information there is."""
    if bid_prev is None or ask_prev is None or not (bid_prev < ask_prev):
        return aggressor_tick_rule(last, last_prev, side_prev)
    if last >= ask_prev:
        return 1
    if last <= bid_prev:
        return -1
    return aggressor_tick_rule(last, last_prev, side_prev)


@dataclass(frozen=True)
class MicroSnapshot:
    """A copy of one symbol's rolling state at `ts`. Immutable by construction."""
    symbol: str
    ts: Optional[datetime]
    buy_vol_15: float
    sell_vol_15: float
    buy_vol_60: float
    sell_vol_60: float
    cvd_15: float
    cvd_60: float
    vol_15: float
    vol_60: float
    sell_share_15: float
    vwap_15: float
    large_vol_15: float
    large_buy_15: float
    large_sell_15: float
    large_print_floor: float
    n_trades_15: int
    imb_ema: float
    spread_ema: float
    bid: Optional[float]
    ask: Optional[float]
    bid_size: float
    ask_size: float
    last: Optional[float]
    minute_close: np.ndarray    # last 60 minute closes (oldest first, NaN where empty)


class SymbolMicro:
    """Per-symbol rings. Single writer (the stream task); readers call
    `snapshot()`, which copies under a short lock."""

    __slots__ = ("symbol", "_lock", "_bins", "_bin_minute", "_head", "_tot15", "_tot60", "_large15",
                 "_quotes", "_qhead", "_qcount", "_imb_ema", "_spr_ema", "_imb_alpha", "_spr_alpha",
                 "_last_quote_ts", "_bid", "_ask", "_bid_size", "_ask_size", "_last", "_last_trade_px",
                 "_side_prev", "_total_volume", "_last_size", "_sizes", "_nsizes", "_large_floor",
                 "_floor_minute", "trades_seen", "_ts", "_session_rows")

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self._lock = threading.Lock()
        self._bins = np.zeros((BINS, _NFIELDS), dtype=np.float64)
        self._bins[:, _HIGH] = np.nan
        self._bins[:, _LOW] = np.nan
        self._bins[:, _CLOSE] = np.nan
        self._bin_minute = np.full(BINS, -1, dtype=np.int64)   # absolute minute index owning each slot
        self._head = -1                                         # absolute minute index of the live bin
        self._tot15 = np.zeros(_NFIELDS, dtype=np.float64)      # running 15-min totals (additive fields)
        self._tot60 = np.zeros(_NFIELDS, dtype=np.float64)
        self._large15 = np.zeros(2, dtype=np.float64)           # large buy / large sell volume, 15 min
        self._quotes = np.zeros((QUOTE_RING_CAPACITY, 5), dtype=np.float64)   # ts, bid, ask, bid_size, ask_size
        self._qhead = 0
        self._qcount = 0
        self._imb_ema = 0.0
        self._spr_ema = 0.0
        self._imb_alpha = 1.0 - 0.5 ** (1.0 / IMB_HALF_LIFE_S)
        self._spr_alpha = 1.0 - 0.5 ** (1.0 / SPR_HALF_LIFE_S)
        self._last_quote_ts: Optional[float] = None
        self._bid: Optional[float] = None
        self._ask: Optional[float] = None
        self._bid_size = 0.0
        self._ask_size = 0.0
        self._last: Optional[float] = None
        self._last_trade_px: Optional[float] = None
        self._side_prev = 0
        self._total_volume: Optional[float] = None
        self._last_size: Optional[float] = None
        self._sizes = np.zeros(LARGE_PRINT_SAMPLE, dtype=np.float64)
        self._nsizes = 0
        self._large_floor = math.inf
        self._floor_minute = -1
        self.trades_seen = 0
        self._ts: Optional[datetime] = None
        self._session_rows: List[np.ndarray] = []            # completed bins kept for the flush

    # ── hot path ────────────────────────────────────────────────────

    def on_tick(self, ts: datetime, *, last: Optional[float], bid: Optional[float], ask: Optional[float],
                bid_size: Optional[float], ask_size: Optional[float], last_size: Optional[float],
                total_volume: Optional[float]) -> None:
        """One Level 1 update. O(1). Never raises (a bad field is dropped)."""
        try:
            with self._lock:
                self._ts = ts
                bid_prev, ask_prev = self._bid, self._ask
                size = self._trade_size(last_size, total_volume)
                if bid is not None and ask is not None:
                    self._apply_quote(ts, float(bid), float(ask), bid_size, ask_size)
                    if bid_prev is None or ask_prev is None:
                        # First update for the symbol: the quote that arrived with
                        # the print is the only quote there is.
                        bid_prev, ask_prev = self._bid, self._ask
                if last is not None:
                    self._last = float(last)
                if size > 0 and last is not None:
                    self._apply_trade(ts, float(last), size, bid_prev, ask_prev)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("micro_book: tick dropped for %s: %s", self.symbol, exc)

    def _trade_size(self, last_size: Optional[float], total_volume: Optional[float]) -> float:
        size = 0.0
        if total_volume is not None:
            tv = float(total_volume)
            if self._total_volume is not None and tv > self._total_volume:
                size = tv - self._total_volume
            elif self._total_volume is None and last_size is not None:
                size = float(last_size)
            self._total_volume = tv
        elif last_size is not None and float(last_size) != (self._last_size or 0.0):
            size = float(last_size)
        if last_size is not None:
            self._last_size = float(last_size)
        return size

    def _apply_quote(self, ts: datetime, bid: float, ask: float, bid_size, ask_size) -> None:
        bs = float(bid_size) if bid_size is not None else self._bid_size
        asz = float(ask_size) if ask_size is not None else self._ask_size
        if not (bid > 0 and ask > bid and bs > 0 and asz > 0):
            return
        self._bid, self._ask, self._bid_size, self._ask_size = bid, ask, bs, asz
        imb = (bs - asz) / (bs + asz)
        spr = (ask - bid) / ((ask + bid) * 0.5)
        if self._qcount == 0:
            self._imb_ema, self._spr_ema = imb, spr
        else:
            self._imb_ema += self._imb_alpha * (imb - self._imb_ema)
            self._spr_ema += self._spr_alpha * (spr - self._spr_ema)
        row = self._quotes[self._qhead]
        row[0], row[1], row[2], row[3], row[4] = ts.timestamp(), bid, ask, bs, asz
        self._qhead = (self._qhead + 1) % QUOTE_RING_CAPACITY
        self._qcount = min(self._qcount + 1, QUOTE_RING_CAPACITY)

    def _apply_trade(self, ts: datetime, px: float, size: float, bid_prev, ask_prev) -> None:
        side = aggressor_quote_rule(px, bid_prev, ask_prev, self._last_trade_px, self._side_prev)
        self._last_trade_px, self._side_prev = px, side
        self.trades_seen += 1
        minute = int(ts.timestamp() // 60)
        self._roll_to(minute)
        b = self._bins[minute % BINS]
        inc = np.zeros(_NFIELDS, dtype=np.float64)
        if side > 0:
            inc[_BUY] = size
            inc[_NBUY] = 1
        elif side < 0:
            inc[_SELL] = size
            inc[_NSELL] = 1
        inc[_VWAPN] = px * size
        inc[_VOL] = size
        inc[_NTR] = 1
        self._note_size(size, minute)
        if size >= self._large_floor:
            inc[_LARGE] = size
            if side > 0:
                self._large15[0] += size
            elif side < 0:
                self._large15[1] += size
        b += inc
        b[_HIGH] = px if np.isnan(b[_HIGH]) else max(b[_HIGH], px)
        b[_LOW] = px if np.isnan(b[_LOW]) else min(b[_LOW], px)
        b[_CLOSE] = px
        self._tot15 += inc
        self._tot60 += inc

    def _roll_to(self, minute: int) -> None:
        """Advance the live bin to `minute`, retiring the bins that leave the
        15- and 60-minute windows from the running totals. Amortised O(1): a
        gap of g minutes retires at most g bins, each exactly once per window."""
        if self._head == minute:
            return
        if self._head < 0:
            self._head = minute
            self._claim(minute)
            return
        gap = minute - self._head
        if gap > BINS:
            # Longer than the ring: every bin has left both windows.
            self._tot15[:] = 0
            self._tot60[:] = 0
            self._large15[:] = 0
            self._bin_minute[:] = -1
            self._head = minute
            self._claim(minute)
            return
        for m in range(self._head + 1, minute + 1):
            self._retire(m - 15, 15)     # window = [m-14, m]: minute m-15 leaves it
            self._retire(m - 60, 60)
            self._claim(m)
        self._head = minute

    def _retire(self, minute: int, window: int) -> None:
        slot = minute % BINS
        if minute < 0 or self._bin_minute[slot] != minute:
            return
        b = self._bins[slot]
        if window == 15:
            self._tot15 -= b
            self._large15[:] = np.maximum(self._large15 - self._split_large(b), 0.0)
        else:
            self._tot60 -= b

    @staticmethod
    def _split_large(b: np.ndarray) -> np.ndarray:
        # large volume is tracked per side only as a 15-min pair; approximate the
        # retiring bin's split by its buy/sell share (exact when one side dominates).
        tot = b[_BUY] + b[_SELL]
        if tot <= 0:
            return np.zeros(2)
        return np.array([b[_LARGE] * b[_BUY] / tot, b[_LARGE] * b[_SELL] / tot])

    def _claim(self, minute: int) -> None:
        slot = minute % BINS
        if self._bin_minute[slot] >= 0 and self._bin_minute[slot] != minute:
            self._session_rows.append(np.concatenate(([self._bin_minute[slot]], self._bins[slot].copy())))
        self._bins[slot] = 0.0
        self._bins[slot, _HIGH] = np.nan
        self._bins[slot, _LOW] = np.nan
        self._bins[slot, _CLOSE] = np.nan
        self._bin_minute[slot] = minute

    def _note_size(self, size: float, minute: int) -> None:
        self._sizes[self._nsizes % LARGE_PRINT_SAMPLE] = size
        self._nsizes += 1
        if minute != self._floor_minute and self._nsizes >= 20:
            n = min(self._nsizes, LARGE_PRINT_SAMPLE)
            self._large_floor = float(np.percentile(self._sizes[:n], LARGE_PRINT_PCTL))
            self._floor_minute = minute

    # ── readers ─────────────────────────────────────────────────────

    @property
    def quote_count(self) -> int:
        return self._qcount

    def snapshot(self) -> MicroSnapshot:
        with self._lock:
            t15 = self._tot15.copy()
            t60 = self._tot60.copy()
            large = self._large15.copy()
            closes = np.full(60, np.nan)
            if self._head >= 0:
                for k in range(60):
                    m = self._head - 59 + k
                    slot = m % BINS
                    if self._bin_minute[slot] == m:
                        closes[k] = self._bins[slot, _CLOSE]
            bid, ask, last = self._bid, self._ask, self._last
            bs, asz = self._bid_size, self._ask_size
            imb, spr, floor, ts = self._imb_ema, self._spr_ema, self._large_floor, self._ts
        vol15 = t15[_VOL]
        sides = t15[_BUY] + t15[_SELL]
        return MicroSnapshot(
            symbol=self.symbol, ts=ts,
            buy_vol_15=float(t15[_BUY]), sell_vol_15=float(t15[_SELL]),
            buy_vol_60=float(t60[_BUY]), sell_vol_60=float(t60[_SELL]),
            cvd_15=float(t15[_BUY] - t15[_SELL]), cvd_60=float(t60[_BUY] - t60[_SELL]),
            vol_15=float(vol15), vol_60=float(t60[_VOL]),
            sell_share_15=float(t15[_SELL] / sides) if sides > 0 else 0.5,
            vwap_15=float(t15[_VWAPN] / vol15) if vol15 > 0 else 0.0,
            large_vol_15=float(t15[_LARGE]), large_buy_15=float(large[0]), large_sell_15=float(large[1]),
            large_print_floor=float(floor) if math.isfinite(floor) else 0.0,
            n_trades_15=int(t15[_NTR]),
            imb_ema=float(imb), spread_ema=float(spr), bid=bid, ask=ask, bid_size=float(bs), ask_size=float(asz),
            last=last, minute_close=closes,
        )

    # ── persistence ─────────────────────────────────────────────────

    def flush(self, root: str, session_date: str) -> str:
        """Write every completed minute bin of the session (plus the live one)
        to `<root>/<SYMBOL>/<session_date>.parquet`, one row per minute, and
        reset the session row buffer. Returns the path."""
        import pandas as pd
        with self._lock:
            rows = list(self._session_rows)
            for slot in range(BINS):
                m = self._bin_minute[slot]
                if m >= 0:
                    rows.append(np.concatenate(([m], self._bins[slot].copy())))
            self._session_rows = []
        rows.sort(key=lambda r: r[0])
        seen = set()
        recs = []
        for r in rows:
            m = int(r[0])
            if m in seen or r[1 + _NTR] <= 0:
                continue
            seen.add(m)
            vol = r[1 + _VOL]
            recs.append({"t_open": datetime.fromtimestamp(m * 60, tz=self._ts.tzinfo if self._ts else None),
                         "buy_vol": r[1 + _BUY], "sell_vol": r[1 + _SELL], "vol": vol,
                         "n_buy": int(r[1 + _NBUY]), "n_sell": int(r[1 + _NSELL]), "n_trades": int(r[1 + _NTR]),
                         "vwap": r[1 + _VWAPN] / vol if vol > 0 else float("nan"),
                         "high": r[1 + _HIGH], "low": r[1 + _LOW], "close": r[1 + _CLOSE], "large_vol": r[1 + _LARGE],
                         "aggressor_rule": "quote"})
        out_dir = os.path.join(root, self.symbol)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{session_date}.parquet")
        pd.DataFrame(recs).to_parquet(path, index=False)
        return path


class MicroBook:
    """Process-wide registry of `SymbolMicro`, installed next to `PriceBook`."""

    _singleton: Optional["MicroBook"] = None

    def __init__(self) -> None:
        self._symbols: Dict[str, SymbolMicro] = {}
        self._lock = threading.Lock()

    @classmethod
    def install(cls, book: "MicroBook") -> None:
        cls._singleton = book

    @classmethod
    def instance(cls) -> Optional["MicroBook"]:
        return cls._singleton

    def _get(self, symbol: str) -> SymbolMicro:
        sym = symbol.upper()
        sm = self._symbols.get(sym)
        if sm is None:
            with self._lock:
                sm = self._symbols.get(sym)
                if sm is None:
                    sm = self._symbols[sym] = SymbolMicro(sym)
        return sm

    def on_tick(self, symbol: str, ts: datetime, **fields) -> None:
        if not symbol:
            return None
        self._get(symbol).on_tick(ts, **fields)
        return None

    def snapshot(self, symbol: str) -> Optional[MicroSnapshot]:
        sm = self._symbols.get((symbol or "").upper())
        return sm.snapshot() if sm is not None else None

    def symbols(self) -> List[str]:
        return list(self._symbols)

    def flush_all(self, root: str, session_date: str) -> List[str]:
        out = []
        for sym, sm in list(self._symbols.items()):
            try:
                if sm.trades_seen > 0:
                    out.append(sm.flush(root, session_date))
            except Exception as exc:
                logger.warning("micro_book: flush failed for %s: %s", sym, exc)
        return out
