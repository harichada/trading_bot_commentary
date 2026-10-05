"""v-micro-book-2026-10-05: Level 1 + tape microstructure book.

Spec: docs/superpowers/specs/2026-10-05-adaptive-rsi-turn-design.md §2.

The Schwab stream delivers last, bid, ask, bid_size, ask_size, last_size and
total_volume for every subscribed name; the price book keeps only prices.
This module keeps the rest, in fixed-size rings, O(1) per update:

* trade detection from the volume delta (quote-only updates carry no trade),
* aggressor tagging by the QUOTE rule (live) — the tick rule exists only for
  the historical archive, which has no quotes,
* a 61-slot minute-bin ring with running 15- and 60-minute totals
  (cumulative volume delta, sell share, VWAP, large prints) that advance on
  EVERY update, so a quiet tape decays to zero instead of looking fresh,
* a 900-slot quote ring with EMA bid/ask-size imbalance and EMA spread,
* array-copy snapshots (with their age) for readers, and an append-safe
  per-session parquet flush.

Honesty about the feed (review round 0, finding 4): Schwab Level 1 is
conflated at ~1 update/s, so one "trade" here is the volume block printed
between two updates, booked at the last price with one side. Block counts are
named `n_updates`, the large-print floor is taken from `last_size` (a real
print size), and the flushed file is tagged `quote_conflated`. A model fitted
on the archive's tick-rule prints must not be scored on these columns as if
they were the same distribution.

Nothing here touches pandas on the hot path. One `SymbolMicro` is ~60 KB; the
book caps itself at `MAX_SYMBOLS` with least-recently-updated eviction.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("TradingBot")

BINS = 61                      # 60 minutes plus one slot so the ring never resizes
QUOTE_RING_CAPACITY = 900      # ~15 min at the observed ~1 update/s per name
IMB_HALF_LIFE_S = 60.0
SPR_HALF_LIFE_S = 300.0
LARGE_PRINT_PCTL = 95.0
LARGE_PRINT_SAMPLE = 2048      # last_size values kept for the session percentile
MAX_SYMBOLS = 128
MAX_SESSION_ROWS = 1600        # > one 04:00-16:05 session (725 min) plus slack; oldest dropped beyond
BLOCK_CAP_MULT = 50.0          # a volume block larger than 50x the median block is suspect
BLOCK_CAP_SESSION_FRAC = 0.05  # ...or larger than 5% of the session's volume so far
WARN_EVERY = 500

# bin field indices — additive fields first (retired by subtraction), then the extrema
(_BUY, _SELL, _VOL, _VWAPN, _LARGE, _LARGE_BUY, _LARGE_SELL, _NUPD, _NBUY, _NSELL, _NTICK, _NQUOTE,
 _UNCLASS, _CAPPED) = range(14)
_ADDITIVE = 14
_HIGH, _LOW, _CLOSE = 14, 15, 16
_NFIELDS = 17


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


def _tagged_by_quote(last: float, bid_prev: Optional[float], ask_prev: Optional[float]) -> bool:
    return bid_prev is not None and ask_prev is not None and bid_prev < ask_prev and (last >= ask_prev or last <= bid_prev)


@dataclass(frozen=True)
class MicroSnapshot:
    """A copy of one symbol's rolling state at `ts`. Immutable by construction.
    `age_s` is how long ago the last update arrived; `head_minute` is the
    epoch minute of the live bin (the windows end there)."""
    symbol: str
    ts: Optional[datetime]
    age_s: float
    head_minute: int
    buy_vol_15: float
    sell_vol_15: float
    unclassified_vol_15: float
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
    n_updates_15: int
    n_quote_tagged_15: int
    n_tick_tagged_15: int
    imb_ema: float
    spread_ema: float
    bid: Optional[float]
    ask: Optional[float]
    bid_size: float
    ask_size: float
    last: Optional[float]
    dropped_updates: int
    capped_vol_15: float        # volume blocks the cap refused to classify
    resyncs: int                # total_volume resets seen this session
    minute_close: np.ndarray    # last 60 minute closes (oldest first, NaN where empty); head_minute is -1 before the first update


class SymbolMicro:
    """Per-symbol rings. Single writer (the stream task); readers call
    `snapshot()`, which copies under a short lock."""

    __slots__ = ("symbol", "_lock", "_bins", "_bin_minute", "_head", "_tot15", "_tot60",
                 "_quotes", "_qhead", "_qcount", "_imb_ema", "_spr_ema", "_imb_alpha", "_spr_alpha",
                 "_bid", "_ask", "_bid_size", "_ask_size", "_last", "_last_trade_px",
                 "_side_prev", "_total_volume", "_last_size", "_resync", "_session_volume",
                 "_blocks", "_nblocks", "_sizes", "_nsizes", "_large_floor", "_floor_minute",
                 "trades_seen", "_ts", "_session_rows", "_flushed_through", "dropped", "_warned",
                 "_tv_max", "resyncs", "_warned_rows")

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol.upper()
        self._lock = threading.Lock()
        self._bins = np.zeros((BINS, _NFIELDS), dtype=np.float64)
        self._bins[:, _HIGH:] = np.nan
        self._bin_minute = np.full(BINS, -1, dtype=np.int64)   # absolute minute index owning each slot
        self._head = -1                                         # absolute minute index of the live bin
        self._tot15 = np.zeros(_ADDITIVE, dtype=np.float64)     # running 15-min totals (additive fields)
        self._tot60 = np.zeros(_ADDITIVE, dtype=np.float64)
        self._quotes = np.zeros((QUOTE_RING_CAPACITY, 5), dtype=np.float64)   # ts, bid, ask, bid_size, ask_size
        self._qhead = 0
        self._qcount = 0
        self._imb_ema = 0.0
        self._spr_ema = 0.0
        self._imb_alpha = 1.0 - 0.5 ** (1.0 / IMB_HALF_LIFE_S)
        self._spr_alpha = 1.0 - 0.5 ** (1.0 / SPR_HALF_LIFE_S)
        self._bid: Optional[float] = None
        self._ask: Optional[float] = None
        self._bid_size = 0.0
        self._ask_size = 0.0
        self._last: Optional[float] = None
        self._last_trade_px: Optional[float] = None
        self._side_prev = 0
        self._total_volume: Optional[float] = None
        self._last_size: Optional[float] = None
        self._resync = False                                    # drop the next delta after a volume reset
        self._tv_max = None                                     # highest total_volume seen (stale duplicates sit below it)
        self.resyncs = 0
        self._warned_rows = 0
        self._session_volume = 0.0
        self._blocks = np.zeros(LARGE_PRINT_SAMPLE, dtype=np.float64)   # recent block sizes (cap statistic)
        self._nblocks = 0
        self._sizes = np.zeros(LARGE_PRINT_SAMPLE, dtype=np.float64)    # recent last_size prints (large floor)
        self._nsizes = 0
        self._large_floor = math.inf
        self._floor_minute = -1
        self.trades_seen = 0
        self._ts: Optional[datetime] = None
        self._session_rows: List[np.ndarray] = []            # completed bins kept for the flush
        self._flushed_through = -1                           # last minute already written by flush()
        self.dropped = 0
        self._warned = 0

    # ── hot path ────────────────────────────────────────────────────

    def on_tick(self, ts: datetime, *, last: Optional[float], bid: Optional[float], ask: Optional[float],
                bid_size: Optional[float], ask_size: Optional[float], last_size: Optional[float],
                total_volume: Optional[float]) -> None:
        """One Level 1 update. O(1). Never raises (a bad update is counted in
        `dropped` and surfaced in the snapshot; the first few are logged)."""
        try:
            with self._lock:
                self._ts = ts
                minute = int(ts.timestamp() // 60)
                self._roll_to(minute)                       # every update advances the windows
                bid_prev, ask_prev = self._bid, self._ask
                size = self._trade_size(last_size, total_volume)
                if bid is not None and ask is not None:
                    self._apply_quote(ts, float(bid), float(ask), bid_size, ask_size)
                    if bid_prev is None or ask_prev is None:
                        bid_prev, ask_prev = self._bid, self._ask    # first update: the only quote there is
                if last is not None:
                    self._last = float(last)
                if size > 0 and last is not None:
                    self._apply_trade(minute, float(last), size, bid_prev, ask_prev)
        except Exception as exc:
            self.dropped += 1
            if self.dropped <= 3 or self.dropped % WARN_EVERY == 0:
                logger.warning("micro_book: update dropped for %s (%d so far): %s", self.symbol, self.dropped, exc)

    def _trade_size(self, last_size: Optional[float], total_volume: Optional[float]) -> float:
        """Volume printed since the previous update, measured against the
        HIGHEST total seen. A total below that high is a stale/out-of-order
        duplicate (size 0, no rebase); a total below half of it is a session
        reset (rebase, count a resync, drop that first delta)."""
        size = 0.0
        if total_volume is not None:
            tv = float(total_volume)
            if self._tv_max is None:
                size = float(last_size) if last_size is not None else 0.0
                self._tv_max = tv
            elif tv > self._tv_max:
                size = 0.0 if self._resync else tv - self._tv_max   # first delta after a reset is unknowable
                self._resync = False
                self._tv_max = tv
            elif tv < 0.5 * self._tv_max:
                self.resyncs += 1
                self._resync = True
                self._tv_max = tv
                size = 0.0
            # else: tv <= max — stale duplicate, nothing printed
            self._total_volume = tv
        elif last_size is not None and float(last_size) != (self._last_size or 0.0):
            size = float(last_size)
        if last_size is not None:
            self._last_size = float(last_size)
            if float(last_size) > 0:
                self._sizes[self._nsizes % LARGE_PRINT_SAMPLE] = float(last_size)
                self._nsizes += 1
        return size

    def _block_is_suspect(self, size: float) -> bool:
        n = min(self._nblocks, LARGE_PRINT_SAMPLE)
        if n >= 30:
            median = float(np.median(self._blocks[:n]))
            if median > 0 and size > BLOCK_CAP_MULT * median and size > BLOCK_CAP_SESSION_FRAC * max(self._session_volume, 1.0):
                return True
        return False

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

    def _apply_trade(self, minute: int, px: float, size: float, bid_prev, ask_prev) -> None:
        suspect = self._block_is_suspect(size)
        if not suspect:
            self._blocks[self._nblocks % LARGE_PRINT_SAMPLE] = size
            self._nblocks += 1
            self._session_volume += size
        self.trades_seen += 1
        b = self._bins[minute % BINS]
        inc = np.zeros(_ADDITIVE, dtype=np.float64)
        inc[_VOL] = size
        inc[_VWAPN] = px * size
        inc[_NUPD] = 1
        if suspect:
            inc[_CAPPED] = size
            if self._warned < 5:
                self._warned += 1
                logger.warning("micro_book: %s volume block %.0f exceeds the cap; counted as unclassified", self.symbol, size)
        else:
            side = aggressor_quote_rule(px, bid_prev, ask_prev, self._last_trade_px, self._side_prev)
            self._last_trade_px, self._side_prev = px, side
            if _tagged_by_quote(px, bid_prev, ask_prev):
                inc[_NQUOTE] = 1
            elif side != 0:
                inc[_NTICK] = 1
            if side > 0:
                inc[_BUY] = size
                inc[_NBUY] = 1
            elif side < 0:
                inc[_SELL] = size
                inc[_NSELL] = 1
            else:
                inc[_UNCLASS] = size
            self._refresh_large_floor(minute)
            last_print = self._last_size or 0.0
            if last_print >= self._large_floor:
                inc[_LARGE] = size
                if side > 0:
                    inc[_LARGE_BUY] = size
                elif side < 0:
                    inc[_LARGE_SELL] = size
        b[:_ADDITIVE] += inc
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
        if self._head < 0 or minute < self._head:
            if minute < self._head:                      # clock went backwards: keep what we have, start over
                for slot in range(BINS):
                    self._stash(slot)
                self._bin_minute[:] = -1
                self._flushed_through = -1
            self._tot15[:] = 0
            self._tot60[:] = 0
            self._head = minute
            self._claim(minute)
            return
        gap = minute - self._head
        if gap > BINS:
            self._tot15[:] = 0
            self._tot60[:] = 0
            for slot in range(BINS):
                self._stash(slot)
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
        b = self._bins[slot, :_ADDITIVE]
        if window == 15:
            self._tot15 -= b
        else:
            self._tot60 -= b

    def _stash(self, slot: int) -> None:
        """Keep a completed bin for the flush (bounded)."""
        m = self._bin_minute[slot]
        if m < 0 or self._bins[slot, _NUPD] <= 0 or m <= self._flushed_through:
            return
        if len(self._session_rows) >= MAX_SESSION_ROWS:
            if self._warned_rows < 3:
                self._warned_rows += 1
                logger.warning("micro_book: %s session rows exceed %d without a flush; oldest dropped", self.symbol, MAX_SESSION_ROWS)
            del self._session_rows[0]
        self._session_rows.append(np.concatenate(([m], self._bins[slot].copy())))

    def _claim(self, minute: int) -> None:
        slot = minute % BINS
        if self._bin_minute[slot] >= 0 and self._bin_minute[slot] != minute:
            self._stash(slot)
        self._bins[slot, :_ADDITIVE] = 0.0
        self._bins[slot, _HIGH:] = np.nan
        self._bin_minute[slot] = minute

    def _refresh_large_floor(self, minute: int) -> None:
        if minute != self._floor_minute and self._nsizes >= 20:
            n = min(self._nsizes, LARGE_PRINT_SAMPLE)
            self._large_floor = float(np.percentile(self._sizes[:n], LARGE_PRINT_PCTL))
            self._floor_minute = minute

    # ── readers ─────────────────────────────────────────────────────

    @property
    def quote_count(self) -> int:
        return self._qcount

    def snapshot(self, now: Optional[datetime] = None) -> MicroSnapshot:
        with self._lock:
            t15 = self._tot15.copy()
            t60 = self._tot60.copy()
            closes = np.full(60, np.nan)
            if self._head >= 0:
                for k in range(60):
                    m = self._head - 59 + k
                    slot = m % BINS
                    if self._bin_minute[slot] == m:
                        closes[k] = self._bins[slot, _CLOSE]
            bid, ask, last = self._bid, self._ask, self._last
            bs, asz = self._bid_size, self._ask_size
            imb, spr, floor, ts, head = self._imb_ema, self._spr_ema, self._large_floor, self._ts, self._head
            dropped, resyncs = self.dropped, self.resyncs
        now = now or datetime.now(timezone.utc)
        age = (now - ts).total_seconds() if ts is not None and ts.tzinfo is not None and now.tzinfo is not None else math.inf
        vol15 = t15[_VOL]
        sides = t15[_BUY] + t15[_SELL]
        return MicroSnapshot(
            symbol=self.symbol, ts=ts, age_s=float(age), head_minute=int(head),
            buy_vol_15=float(t15[_BUY]), sell_vol_15=float(t15[_SELL]), unclassified_vol_15=float(t15[_UNCLASS]),
            buy_vol_60=float(t60[_BUY]), sell_vol_60=float(t60[_SELL]),
            cvd_15=float(t15[_BUY] - t15[_SELL]), cvd_60=float(t60[_BUY] - t60[_SELL]),
            vol_15=float(vol15), vol_60=float(t60[_VOL]),
            sell_share_15=float(t15[_SELL] / sides) if sides > 0 else 0.5,
            vwap_15=float(t15[_VWAPN] / vol15) if vol15 > 0 else 0.0,
            large_vol_15=float(t15[_LARGE]), large_buy_15=float(t15[_LARGE_BUY]), large_sell_15=float(t15[_LARGE_SELL]),
            large_print_floor=float(floor) if math.isfinite(floor) else 0.0,
            n_updates_15=int(t15[_NUPD]), n_quote_tagged_15=int(t15[_NQUOTE]), n_tick_tagged_15=int(t15[_NTICK]),
            imb_ema=float(imb), spread_ema=float(spr), bid=bid, ask=ask, bid_size=float(bs), ask_size=float(asz),
            last=last, dropped_updates=int(dropped), capped_vol_15=float(t15[_CAPPED]), resyncs=int(resyncs),
            minute_close=closes,
        )

    # ── persistence ─────────────────────────────────────────────────

    def flush(self, root: str, session_date: str) -> str:
        """Append every completed minute bin not yet written to
        `<root>/<SYMBOL>/<date>.parquet` (one row per minute, UTC `t_open`),
        one file per ET session date the rows belong to, merging with rows
        already on disk (the fuller row wins), via a temp file and rename.
        The live minute is written but not marked final, so the rest of it
        lands on the next flush. Rows are released only after the write
        succeeded. Returns the path for `session_date`."""
        import pandas as pd
        with self._lock:
            rows = list(self._session_rows)
            head = self._head
            for slot in range(BINS):
                m = self._bin_minute[slot]
                if m > self._flushed_through and self._bins[slot, _NUPD] > 0:
                    rows.append(np.concatenate(([m], self._bins[slot].copy())))
            flushed_through = self._flushed_through
        recs = {}
        for r in rows:
            m = int(r[0])
            if m <= flushed_through:
                continue
            if m not in recs or r[1 + _NUPD] > recs[m][1 + _NUPD]:
                recs[m] = r
        final_through = max([m for m in recs if m != head], default=flushed_through)
        by_date: dict = {}
        for m, r in recs.items():
            t_utc = datetime.fromtimestamp(m * 60, tz=timezone.utc)
            et_date = _et_date(t_utc)
            by_date.setdefault(et_date, []).append(_rec(t_utc, r[1:]))
        out_dir = os.path.join(root, self.symbol)
        os.makedirs(out_dir, exist_ok=True)
        target = os.path.join(out_dir, f"{session_date}.parquet")
        if session_date not in by_date and not os.path.exists(target):
            by_date[session_date] = []
        for d, recs_d in by_date.items():
            path = os.path.join(out_dir, f"{d}.parquet")
            new = pd.DataFrame(recs_d)
            if os.path.exists(path):
                try:
                    old = pd.read_parquet(path)
                    new = pd.concat([old, new], ignore_index=True) if not new.empty else old
                    if not new.empty:
                        new = (new.sort_values(["t_open", "n_updates"]).drop_duplicates(subset="t_open", keep="last")
                               .sort_values("t_open").reset_index(drop=True))
                except Exception as exc:
                    logger.warning("micro_book: could not merge existing %s: %s", path, exc)
            tmp = path + ".tmp"
            new.to_parquet(tmp, index=False)
            os.replace(tmp, path)
        with self._lock:
            self._session_rows = [r for r in self._session_rows if int(r[0]) > final_through]
            self._flushed_through = max(self._flushed_through, final_through)
        return target


def _et_date(t_utc: datetime) -> str:
    try:
        from zoneinfo import ZoneInfo
        return t_utc.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return t_utc.strftime("%Y-%m-%d")


def _rec(t_utc: datetime, f: np.ndarray) -> dict:
    vol = f[_VOL]
    return {"t_open": t_utc,
            "buy_vol": f[_BUY], "sell_vol": f[_SELL], "unclassified_vol": f[_UNCLASS], "capped_vol": f[_CAPPED],
            "vol": vol, "n_buy": int(f[_NBUY]), "n_sell": int(f[_NSELL]), "n_updates": int(f[_NUPD]),
            "n_quote_tagged": int(f[_NQUOTE]), "n_tick_tagged": int(f[_NTICK]),
            "vwap": f[_VWAPN] / vol if vol > 0 else float("nan"),
            "high": f[_HIGH], "low": f[_LOW], "close": f[_CLOSE],
            "large_vol": f[_LARGE], "large_buy": f[_LARGE_BUY], "large_sell": f[_LARGE_SELL],
            "aggressor_rule": "quote_conflated"}


class MicroBook:
    """Process-wide registry of `SymbolMicro`, installed next to `PriceBook`.
    Bounded at MAX_SYMBOLS: the least recently updated symbol is evicted."""

    _singleton: Optional["MicroBook"] = None

    def __init__(self, max_symbols: int = MAX_SYMBOLS) -> None:
        self._symbols: Dict[str, SymbolMicro] = {}
        self._lock = threading.Lock()
        self._max = max(1, int(max_symbols))

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
                    if len(self._symbols) >= self._max:
                        self._evict_oldest()
                    sm = self._symbols[sym] = SymbolMicro(sym)
        return sm

    def _evict_oldest(self) -> None:
        def stamp(item):
            t = item[1]._ts
            return t.timestamp() if t is not None else -1.0
        victim, sm = min(self._symbols.items(), key=stamp)
        logger.info("micro_book: evicting %s (cap %d symbols, %d unflushed rows)", victim, self._max,
                    len(sm._session_rows))
        del self._symbols[victim]

    def on_tick(self, symbol: str, ts: datetime, **fields) -> None:
        if not symbol:
            return None
        self._get(symbol).on_tick(ts, **fields)
        return None

    def snapshot(self, symbol: str, now: Optional[datetime] = None) -> Optional[MicroSnapshot]:
        sm = self._symbols.get((symbol or "").upper())
        return sm.snapshot(now) if sm is not None else None

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
