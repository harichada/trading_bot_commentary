"""SPEC §6 schedule (ET): 16:20-16:30 EOD scan -> signals; next open -> shadow fills;
daily EOD -> stop/trail/time exits. SHADOW ONLY: writes log rows, never orders."""
from __future__ import annotations

import importlib
import logging
import uuid
from datetime import datetime, time
from typing import Callable, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from core.swing_lane import config as C
from core.swing_lane import data as D
from core.swing_lane import indicators as I
from core.swing_lane import risk as R
from core.swing_lane.shadow_log import ShadowLog, make_row

logger = logging.getLogger("swing_lane")
ET = ZoneInfo("America/New_York")


def in_scan_window(now_et: datetime) -> bool:
    if now_et.tzinfo is not None:
        now_et = now_et.astimezone(ET)
    if now_et.weekday() >= 5:
        return False
    (h0, m0), (h1, m1) = C.SCAN_WINDOW
    return time(h0, m0) <= now_et.time().replace(second=0, microsecond=0) <= time(h1, m1)


def _enabled_setups(settings) -> dict:
    """letter -> imported module, for enabled + implemented setups only."""
    mods = {}
    for k, (flag, modname) in C.SETUPS.items():
        if not settings.setups.get(k):
            continue
        if modname is None:
            logger.warning("swing_lane: setup %s (%s) is research-only (failed backtest); not run", k, flag)
            continue
        mods[k] = importlib.import_module(f"core.swing_lane.setups.{modname}")
    return mods


def _open_positions(rows: list) -> dict:
    """key -> latest fill row, for shadow fills that have no exit row yet."""
    def key(r):
        return (r["setup"], r["sym"], r["signal_date"])
    fills = {key(r): r for r in rows if r.get("event") == "fill"}
    for r in rows:
        if r.get("event") == "exit":
            fills.pop(key(r), None)
    return fills


def run_eod_scan(settings, now_et: datetime, *, held_symbols: Iterable[str] = (),
                 realized_today: Optional[float] = None,
                 bars_loader: Callable = None, earnings_loader: Callable = None,
                 universe: Optional[list] = None, params: Optional[dict] = None) -> list:
    """Run the EOD scan; returns the rows appended to the shadow log."""
    if settings.live:
        raise RuntimeError("swing_lane has no live/order path")
    if not settings.shadow:
        return []
    if not in_scan_window(now_et):
        logger.info("swing_lane: outside scan window %s; skipping", C.SCAN_WINDOW)
        return []
    setups = _enabled_setups(settings)
    if not setups:
        return []
    bars_loader = bars_loader or (lambda s: D.load_daily(settings.data_root, s))
    earnings_loader = earnings_loader or (lambda s: D.load_earnings(settings.data_root, s))
    params = params or C.load_params()
    phash = C.params_hash(params)
    universe = universe if universe is not None else C.load_universe()
    deny = frozenset(s.upper() for s in settings.denylist)
    held = {s.upper() for s in held_symbols}
    log = ShadowLog(settings.log_dir)
    run_id = uuid.uuid4().hex[:12]
    base = dict(run_id=run_id, code_version=C.CODE_VERSION, params_hash=phash)
    out = []

    def emit(**kw):
        row = make_row(**base, **kw)
        log.append(row)
        out.append(row)

    spy = bars_loader("SPY")
    spy = D.drop_partial(spy, now_et) if spy is not None else None
    if spy is None or len(spy) < C.MIN_BARS or pd.Timestamp(spy.index[-1]).date() != now_et.date():
        emit(event="run_skip", skipped_reason="spy_bars_missing_or_stale")
        return out
    regime = I.regime_on(spy)
    asof = spy.index[-1]
    cands = []
    for sym in universe:
        if sym in C.REGIME_ONLY:
            continue
        if sym in deny:
            emit(event="skip", sym=sym, skipped_reason="hands_off_denylist")
            continue
        if sym in held:
            emit(event="skip", sym=sym, skipped_reason="existing_position")
            continue
        try:
            df = bars_loader(sym)
            df = D.drop_partial(df, now_et) if df is not None else None
            if df is None or len(df) < C.MIN_BARS or df.index[-1] != asof:
                emit(event="skip", sym=sym, skipped_reason="bars_missing_or_stale")
                continue
            ind = I.compute(df)
            earn = earnings_loader(sym)
            for k, mod in setups.items():
                kwargs = {"earnings": earn} if k == "C" else {}
                for sig in mod.scan(df, ind, params[k], sym=sym, **kwargs):
                    cands.append((sig, earn))
        except Exception as exc:  # never escape into the bot
            logger.warning("swing_lane: %s scan error: %s", sym, exc)
            emit(event="skip", sym=sym, skipped_reason=f"scan_error:{type(exc).__name__}")

    rows = log.read()
    open_pos = _open_positions(rows)
    open_syms = {k[1] for k in open_pos}
    n_open = len(open_pos)
    open_risk = sum(float(r.get("planned_risk_usd") or 0) for r in open_pos.values())
    loss_block = R.daily_loss_blocked(realized_today)
    for k in sorted({s.setup for s, _ in cands}):
        ranked = sorted([c for c in cands if c[0].setup == k], key=lambda c: -c[0].score)
        for rank, (sig, earn) in enumerate(ranked, 1):
            d = pd.Timestamp(sig.signal_date).date()
            blocked, why, enext = R.earnings_check(d, earn, settings.earnings_blackout)
            sh = R.shares(sig.close, sig.planned_stop)
            risk_usd = round(sh * (sig.close - sig.planned_stop), 2)
            slot = R.slot_available(n_open, open_risk, risk_usd)
            reason = None
            if not regime:
                reason = "regime_off"
            elif why:
                reason = why
                logger.info("swing_lane: %s %s no entry: %s", k, sig.sym, why)
            elif sig.sym in open_syms:
                reason = "one_per_symbol"
            elif loss_block:
                reason = "daily_loss_limit"
            elif sh <= 0:
                reason = "zero_shares"
            elif not slot:
                reason = "no_slot"
            emit(event="signal", setup=k, sym=sig.sym, signal_date=d, regime_on=regime,
                 close=sig.close, atr14=sig.atr14, rvol=sig.rvol, gap=sig.gap, rng20_atr=sig.rng20_atr,
                 earnings_next=enext, blackout_skip=why in ("earnings_blackout", "earnings_data_missing"),
                 rank=rank, slot_available=slot, planned_entry="open", planned_stop=sig.planned_stop,
                 planned_target=sig.planned_target, planned_shares=sh, planned_risk_usd=risk_usd,
                 skipped_reason=reason)
            if reason is None:
                n_open += 1
                open_risk += risk_usd
                open_syms.add(sig.sym)
    return out


def record_fills(settings, today, bars_loader: Callable = None) -> list:
    """Shadow: record the actual open of `today` as the fill for yesterday's accepted signals."""
    if settings.live or not settings.shadow:
        return []
    bars_loader = bars_loader or (lambda s: D.load_daily(settings.data_root, s))
    log = ShadowLog(settings.log_dir)
    rows = log.read()
    done = {(r["setup"], r["sym"], r["signal_date"]) for r in rows if r.get("event") == "fill"}
    out = []
    for r in rows:
        if r.get("event") != "signal" or r.get("skipped_reason") is not None:
            continue
        key = (r["setup"], r["sym"], r["signal_date"])
        if key in done or str(today) <= r["signal_date"]:
            continue
        df = bars_loader(r["sym"])
        if df is None:
            continue
        nxt = [i for i in df.index if pd.Timestamp(i).date().isoformat() > r["signal_date"]]
        if not nxt:
            continue
        fill = float(df.loc[nxt[0], "open"])
        stop = float(r["planned_stop"])
        sh = R.shares(fill, stop)  # recompute from actual fill (SPEC §4)
        tgt = (fill + 3 * (fill - stop)) if r["setup"] == "C" else None
        row = make_row(**{k: r[k] for k in ("run_id", "setup", "sym", "signal_date", "code_version", "params_hash")},
                       event="fill", fill_date=pd.Timestamp(nxt[0]).date(), fill_price=fill,
                       slippage_bps=round((fill / r["close"] - 1) * 1e4, 2), planned_stop=stop,
                       planned_target=tgt, planned_shares=sh,
                       planned_risk_usd=round(sh * (fill - stop), 2), skipped_reason=None if sh > 0 else "zero_shares")
        if sh > 0:
            log.append(row)
            out.append(row)
    return out


def evaluate_exit(setup: str, fill: float, stop0: float, target, bars: pd.DataFrame, atr: pd.Series,
                  earnings: Optional[set] = None):
    """Walk daily bars from the fill day; returns dict for the exit row or None if still open.
    bars.index[0] must be the fill day (hold day 1)."""
    stop, hi_close, updates = stop0, -float("inf"), []
    r1 = fill - stop0
    mae = mfe = 0.0
    for day, (ix, b) in enumerate(bars.iterrows(), 1):
        if day > 1 and b.open <= stop:
            px, why, gt = float(b.open), "gapstop", True
        elif b.low <= stop:
            px, why, gt = stop, ("trail" if stop > stop0 else "stop"), False
        elif target is not None and b.high >= target:
            px, why, gt = float(target), "target", False
        else:
            px = None
        mae = min(mae, (b.low - fill) / r1); mfe = max(mfe, (b.high - fill) / r1)
        if px is None and R.time_stop_due(day):
            px, why, gt = float(b.close), "time", False
        if px is None and earnings:
            nd = R.next_trading_days(pd.Timestamp(ix).date(), 1)[0]
            if nd in earnings:
                px, why, gt = float(b.close), "earnings", False
        if px is not None:
            return dict(exit_date=pd.Timestamp(ix).date(), exit_price=px, exit_reason=why, gap_through=gt,
                        R=round((px - fill) / r1, 3), mae_R=round(mae, 3), mfe_R=round(mfe, 3),
                        hold_days=day, stop_updates=updates)
        hi_close = max(hi_close, float(b.close))
        if setup == "B":
            ns = R.b_trail_stop(stop, fill, hi_close, float(atr.loc[ix]), float(b.close))
            if ns > stop:
                stop = ns
                updates.append({"date": pd.Timestamp(ix).date().isoformat(), "stop": round(ns, 4)})
    return None


def update_exits(settings, bars_loader: Callable = None, earnings_loader: Callable = None) -> list:
    """Daily EOD: evaluate stops/trails/time/earnings exits for open shadow positions."""
    if settings.live or not settings.shadow:
        return []
    bars_loader = bars_loader or (lambda s: D.load_daily(settings.data_root, s))
    earnings_loader = earnings_loader or (lambda s: D.load_earnings(settings.data_root, s))
    log = ShadowLog(settings.log_dir)
    out = []
    for key, f in _open_positions(log.read()).items():
        try:
            df = bars_loader(f["sym"])
            if df is None:
                continue
            atr = I.compute(df)["atr"]
            df = df[[pd.Timestamp(i).date().isoformat() >= f["fill_date"] for i in df.index]]
            res = evaluate_exit(f["setup"], f["fill_price"], f["planned_stop"], f["planned_target"],
                                df, atr, earnings_loader(f["sym"]))
            if res is None:
                continue
            res["pnl_usd"] = round((res["exit_price"] - f["fill_price"]) * f["planned_shares"], 2)
            row = make_row(**{k: f[k] for k in ("run_id", "setup", "sym", "signal_date", "fill_date",
                                                 "fill_price", "planned_shares", "planned_stop",
                                                 "code_version", "params_hash")}, event="exit", **res)
            log.append(row)
            out.append(row)
        except Exception as exc:
            logger.warning("swing_lane: exit eval %s failed: %s", key, exc)
    return out
