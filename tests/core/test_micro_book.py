"""v-micro-book-2026-10-05: the Level 1 + tape microstructure book
(docs/superpowers/specs/2026-10-05-adaptive-rsi-turn-design.md §2).

Synthetic tick streams only; no network, no engine."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from core.micro_book import (MicroBook, QUOTE_RING_CAPACITY, SymbolMicro, aggressor_quote_rule,
                             aggressor_tick_rule)

ET = timezone(timedelta(hours=-4))
T0 = datetime(2026, 10, 5, 10, 0, 0, tzinfo=ET)


def _ts(seconds: float) -> datetime:
    return T0 + timedelta(seconds=seconds)


# ── aggressor tagging ─────────────────────────────────────────────────

def test_quote_rule_buy_sell_inside_and_flat():
    assert aggressor_quote_rule(last=100.05, bid_prev=100.0, ask_prev=100.05, last_prev=100.02, side_prev=-1) == 1
    assert aggressor_quote_rule(last=100.0, bid_prev=100.0, ask_prev=100.05, last_prev=100.02, side_prev=1) == -1
    # inside the spread: tick direction against the previous trade
    assert aggressor_quote_rule(last=100.03, bid_prev=100.0, ask_prev=100.05, last_prev=100.02, side_prev=-1) == 1
    assert aggressor_quote_rule(last=100.01, bid_prev=100.0, ask_prev=100.05, last_prev=100.02, side_prev=1) == -1
    # flat inside the spread inherits
    assert aggressor_quote_rule(last=100.02, bid_prev=100.0, ask_prev=100.05, last_prev=100.02, side_prev=-1) == -1
    # no quote known yet: falls back to the tick rule, never raises
    assert aggressor_quote_rule(last=100.02, bid_prev=None, ask_prev=None, last_prev=100.0, side_prev=0) == 1


def test_tick_rule_sign_and_inheritance():
    assert aggressor_tick_rule(101.0, 100.0, 0) == 1
    assert aggressor_tick_rule(99.0, 100.0, 0) == -1
    assert aggressor_tick_rule(100.0, 100.0, -1) == -1
    assert aggressor_tick_rule(100.0, None, 0) == 0


# ── trade detection + bins ────────────────────────────────────────────

def test_trade_detected_from_total_volume_delta_and_quote_only_updates_ignored():
    sm = SymbolMicro("MU")
    sm.on_tick(_ts(0), last=None, bid=100.0, ask=100.05, bid_size=500, ask_size=300, last_size=None, total_volume=None)
    assert sm.trades_seen == 0
    sm.on_tick(_ts(1), last=100.05, bid=100.0, ask=100.05, bid_size=500, ask_size=200, last_size=100, total_volume=1_000_100)
    assert sm.trades_seen == 1
    sm.on_tick(_ts(2), last=100.05, bid=100.0, ask=100.06, bid_size=500, ask_size=200, last_size=100, total_volume=1_000_100)
    assert sm.trades_seen == 1          # volume unchanged, same last_size → quote-only
    sm.on_tick(_ts(3), last=100.0, bid=100.0, ask=100.06, bid_size=500, ask_size=200, last_size=250, total_volume=1_000_350)
    snap = sm.snapshot()
    assert sm.trades_seen == 2
    assert snap.buy_vol_15 == 100 and snap.sell_vol_15 == 250    # first hit the ask, second hit the bid
    assert snap.cvd_15 == -150 and snap.cvd_60 == -150


def _stream(sm: SymbolMicro, minutes: int, seed: int = 1):
    """A synthetic session: ~2 updates/s, half of them trades. Returns the
    (ts, size, side) list for brute-force comparison."""
    rng = np.random.default_rng(seed)
    bid, ask = 100.0, 100.05
    vol = 0
    log = []
    last_prev, side_prev = 100.02, 0
    for i in range(minutes * 120):
        t = _ts(i * 0.5)
        if rng.random() < 0.5:
            size = int(rng.choice([1, 50, 100, 200, 800]))
            px = ask if rng.random() < 0.5 else bid
            if rng.random() < 0.2:
                px = round(bid + 0.02, 2)
            side = aggressor_quote_rule(px, bid, ask, last_prev, side_prev)
            vol += size
            sm.on_tick(t, last=px, bid=bid, ask=ask, bid_size=400, ask_size=300, last_size=size, total_volume=vol)
            log.append((t, size, side, px))
            last_prev, side_prev = px, side
        else:
            bid = round(bid + rng.choice([-0.01, 0, 0.01]), 2)
            ask = round(bid + 0.05, 2)
            sm.on_tick(t, last=None, bid=bid, ask=ask, bid_size=int(rng.integers(100, 900)),
                       ask_size=int(rng.integers(100, 900)), last_size=None, total_volume=None)
    return log


def test_window_totals_match_brute_force_after_bin_rolls():
    sm = SymbolMicro("MU")
    log = _stream(sm, minutes=90)
    snap = sm.snapshot()
    now = log[-1][0]
    def brute(window_min):
        lo = now.replace(second=0, microsecond=0) - timedelta(minutes=window_min - 1)
        rows = [r for r in log if r[0] >= lo]
        buy = sum(s for _, s, side, _ in rows if side > 0)
        sell = sum(s for _, s, side, _ in rows if side < 0)
        return buy, sell
    b15, s15 = brute(15)
    b60, s60 = brute(60)
    assert (snap.buy_vol_15, snap.sell_vol_15) == (b15, s15)
    assert (snap.buy_vol_60, snap.sell_vol_60) == (b60, s60)
    assert snap.cvd_15 == b15 - s15 and snap.cvd_60 == b60 - s60
    assert 0.0 <= snap.sell_share_15 <= 1.0
    assert snap.vwap_15 > 0 and 99.0 < snap.vwap_15 < 101.0


def test_large_print_floor_tracks_the_session_and_flags_big_prints():
    sm = SymbolMicro("MU")
    _stream(sm, minutes=20)
    snap = sm.snapshot()
    assert snap.large_print_floor >= 200           # 95th percentile of {1,50,100,200,800}
    assert snap.large_vol_15 > 0 and snap.large_vol_15 <= snap.vol_15
    assert snap.large_buy_15 + snap.large_sell_15 <= snap.large_vol_15 + 1e-9


# ── quote ring + EMAs ─────────────────────────────────────────────────

def test_quote_ring_is_bounded_and_imbalance_ema_in_range():
    sm = SymbolMicro("MU")
    for i in range(QUOTE_RING_CAPACITY * 3):
        sm.on_tick(_ts(i), last=None, bid=100.0, ask=100.05, bid_size=900, ask_size=100, last_size=None, total_volume=None)
    snap = sm.snapshot()
    assert sm.quote_count == QUOTE_RING_CAPACITY
    assert 0.7 < snap.imb_ema <= 1.0 and snap.spread_ema > 0
    for i in range(300):
        sm.on_tick(_ts(10_000 + i), last=None, bid=100.0, ask=100.05, bid_size=100, ask_size=900, last_size=None, total_volume=None)
    assert -1.0 <= sm.snapshot().imb_ema < -0.5


def test_crossed_or_empty_quotes_are_dropped():
    sm = SymbolMicro("MU")
    sm.on_tick(_ts(0), last=None, bid=100.0, ask=100.05, bid_size=500, ask_size=500, last_size=None, total_volume=None)
    sm.on_tick(_ts(1), last=None, bid=100.10, ask=100.05, bid_size=500, ask_size=500, last_size=None, total_volume=None)  # crossed
    sm.on_tick(_ts(2), last=None, bid=100.0, ask=100.05, bid_size=0, ask_size=500, last_size=None, total_volume=None)     # empty side
    assert sm.quote_count == 1


# ── book, snapshots, flush ────────────────────────────────────────────

def test_snapshot_is_a_copy_and_book_routes_symbols():
    book = MicroBook()
    MicroBook.install(book)
    assert MicroBook.instance() is book
    book.on_tick("mu", _ts(0), last=100.05, bid=100.0, ask=100.05, bid_size=1, ask_size=1, last_size=10, total_volume=10)
    book.on_tick("TSLA", _ts(0), last=None, bid=50.0, ask=50.1, bid_size=1, ask_size=1, last_size=None, total_volume=None)
    s1 = book.snapshot("MU")
    book.on_tick("MU", _ts(1), last=100.0, bid=100.0, ask=100.05, bid_size=1, ask_size=1, last_size=10, total_volume=20)
    assert s1.buy_vol_15 == 10 and book.snapshot("MU").sell_vol_15 == 10     # the earlier snapshot did not change
    assert book.snapshot("NOPE") is None and sorted(book.symbols()) == ["MU", "TSLA"]
    assert book.on_tick("", _ts(2), last=1.0, bid=None, ask=None, bid_size=None, ask_size=None, last_size=None, total_volume=None) is None


def test_flush_writes_one_row_per_minute(tmp_path):
    sm = SymbolMicro("MU")
    _stream(sm, minutes=7)
    path = sm.flush(str(tmp_path), session_date="2026-10-05")
    import pandas as pd
    df = pd.read_parquet(path)
    assert path.endswith("MU/2026-10-05.parquet")
    assert 6 <= len(df) <= 8
    for col in ("t_open", "buy_vol", "sell_vol", "vol", "vwap", "high", "low", "close", "large_vol", "n_trades"):
        assert col in df.columns
    assert (df["buy_vol"] + df["sell_vol"] == df["vol"]).all()


def test_hot_path_is_fast_enough():
    sm = SymbolMicro("MU")
    t0 = time.perf_counter()
    _stream(sm, minutes=120)       # ~14k updates
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"14k updates took {elapsed:.2f}s"


def test_stream_handler_feeds_the_micro_book(monkeypatch):
    """The stream parse site passes sizes and volume to the installed book."""
    import inspect
    import core.schwab_stream as ss
    src = inspect.getsource(ss)
    assert '_field(record, "BID_SIZE", "4")' in src and '_field(record, "TOTAL_VOLUME", "8")' in src
    assert 'last=_f(print_px)' in src and '_field(record, "LAST_PRICE", "3")' in src
    assert ss._field({"BID_SIZE": None, "4": 7}, "BID_SIZE", "4") == 7
    assert "mb.on_tick(" in src and "MicroBook.instance()" in src
    assert ss._f("12.5") == 12.5 and ss._f(None) is None and ss._f("x") is None


def test_desk_loop_flushes_the_book_at_the_journal_step(tmp_path, monkeypatch):
    import asyncio as _a
    from types import SimpleNamespace
    from core.loops import desk_loop as dl
    book = MicroBook()
    MicroBook.install(book)
    book.on_tick("MU", _ts(0), last=100.05, bid=100.0, ask=100.05, bid_size=1, ask_size=1, last_size=10, total_volume=10)
    eng = SimpleNamespace(_audit=lambda *a, **k: None)
    loop = dl.DeskLoop(eng, config_fn=lambda: SimpleNamespace(MICRO_DATA_DIR=str(tmp_path)))
    _a.run(loop._flush_micro(SimpleNamespace(MICRO_DATA_DIR=str(tmp_path)), "2026-10-05"))
    assert (tmp_path / "MU" / "2026-10-05.parquet").exists()
    # a second flush of the same day is harmless (rows already persisted are not duplicated)
    _a.run(loop._flush_micro(SimpleNamespace(MICRO_DATA_DIR=str(tmp_path)), "2026-10-05"))


# ── review round 0 findings ──────────────────────────────────────────

def test_windows_decay_during_quote_only_updates():
    sm = SymbolMicro("MU")
    sm.on_tick(_ts(0), last=100.05, bid=100.0, ask=100.05, bid_size=500, ask_size=300, last_size=500, total_volume=500)
    assert sm.snapshot(_ts(1)).vol_15 == 500
    for i in range(1, 16 * 60):
        sm.on_tick(_ts(i), last=None, bid=100.0, ask=100.05, bid_size=500, ask_size=300, last_size=None, total_volume=None)
    snap = sm.snapshot(_ts(16 * 60))
    assert snap.vol_15 == 0 and snap.cvd_15 == 0 and snap.vol_60 == 500 and snap.age_s == pytest.approx(1.0)
    for i in range(16 * 60, 61 * 60):
        sm.on_tick(_ts(i), last=None, bid=100.0, ask=100.05, bid_size=500, ask_size=300, last_size=None, total_volume=None)
    assert sm.snapshot().vol_60 == 0


def test_volume_reset_does_not_book_a_phantom_print():
    sm = SymbolMicro("MU")
    q = dict(bid=100.0, ask=100.05, bid_size=500, ask_size=300)
    sm.on_tick(_ts(0), last=100.05, last_size=100, total_volume=1_000_000, **q)
    sm.on_tick(_ts(1), last=100.05, last_size=200, total_volume=1_000_200, **q)
    sm.on_tick(_ts(2), last=100.05, last_size=100, total_volume=500, **q)             # reset / out of order
    sm.on_tick(_ts(3), last=100.05, last_size=400, total_volume=1_000_400, **q)       # first delta after reset: dropped
    sm.on_tick(_ts(4), last=100.05, last_size=100, total_volume=1_000_500, **q)
    assert sm.snapshot().vol_15 == 100 + 200 + 100


def test_implausible_block_is_unclassified_not_a_side():
    sm = SymbolMicro("MU")
    q = dict(bid=100.0, ask=100.05, bid_size=500, ask_size=300)
    vol = 0
    for i in range(60):
        vol += 100
        sm.on_tick(_ts(i), last=100.05, last_size=100, total_volume=vol, **q)
    vol += 5_000_000
    sm.on_tick(_ts(61), last=100.05, last_size=100, total_volume=vol, **q)
    snap = sm.snapshot()
    assert snap.unclassified_vol_15 == 5_000_000 and snap.buy_vol_15 == 6000
    assert snap.buy_vol_15 + snap.sell_vol_15 + snap.unclassified_vol_15 == snap.vol_15


def test_first_print_uses_quote_in_the_same_update_and_side_zero_is_accounted():
    sm = SymbolMicro("MU")
    sm.on_tick(_ts(0), last=100.02, bid=None, ask=None, bid_size=None, ask_size=None, last_size=50, total_volume=50)
    snap = sm.snapshot()
    assert snap.vol_15 == 50 and snap.unclassified_vol_15 == 50 and snap.n_quote_tagged_15 == 0
    sm.on_tick(_ts(1), last=100.05, bid=100.0, ask=100.05, bid_size=10, ask_size=10, last_size=50, total_volume=100)
    snap = sm.snapshot()
    assert snap.buy_vol_15 == 50 and snap.n_quote_tagged_15 == 1


def test_flush_twice_only_adds_rows_and_never_reemits_old_bins(tmp_path):
    import pandas as pd
    sm = SymbolMicro("MU")
    log = _stream(sm, minutes=7)
    p1 = sm.flush(str(tmp_path), "2026-10-05")
    n1 = len(pd.read_parquet(p1))
    # a restart-like second flush with two more minutes of ticks
    t_last = log[-1][0]
    vol = sm._total_volume
    for k in range(240):
        vol += 10
        sm.on_tick(t_last + timedelta(seconds=1 + k * 0.5), last=100.0, bid=99.95, ask=100.0, bid_size=5, ask_size=5, last_size=10, total_volume=vol)
    p2 = sm.flush(str(tmp_path), "2026-10-05")
    df = pd.read_parquet(p2)
    assert p1 == p2 and len(df) >= n1 and df["t_open"].is_unique
    assert str(df["t_open"].dt.tz) == "UTC" and (df["aggressor_rule"] == "quote_conflated").all()
    # next day's file must not contain yesterday's bins
    p3 = sm.flush(str(tmp_path), "2026-10-06")
    assert len(pd.read_parquet(p3)) == 0


def test_book_caps_symbols_with_eviction():
    book = MicroBook(max_symbols=3)
    for i, s in enumerate(["A", "B", "C"]):
        book.on_tick(s, _ts(i), last=None, bid=1.0, ask=1.01, bid_size=1, ask_size=1, last_size=None, total_volume=None)
    book.on_tick("D", _ts(10), last=None, bid=1.0, ask=1.01, bid_size=1, ask_size=1, last_size=None, total_volume=None)
    assert sorted(book.symbols()) == ["B", "C", "D"]


def test_concurrent_writer_and_reader_stay_consistent():
    import threading
    sm = SymbolMicro("MU")
    stop = threading.Event()
    bad = []
    def reader():
        while not stop.is_set():
            s = sm.snapshot()
            if not (s.buy_vol_15 + s.sell_vol_15 + s.unclassified_vol_15 <= s.vol_15 + 1e-6):
                bad.append(s)
    t = threading.Thread(target=reader); t.start()
    _stream(sm, minutes=30)
    stop.set(); t.join()
    assert not bad
