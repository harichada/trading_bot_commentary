"""Weekly trading-bot performance report — measurement-driven.

v-weekly-report-2026-05-22: this is the "stop patching, measure honestly"
artifact. Builds slices across strategy / symbol-class / time-of-day /
classifier reading, computes realized P&L per trade, and (critically)
simulates counterfactual short trades for every `rule_allowed=['short']`
classifier signal that the bot saw but never traded.

Output: docs/weekly_report_<YYYY-MM-DD>.md

Inputs:
  - trading_bot.log + trading_bot.log.1..N (audit lines for this week)
  - Schwab orders for the same period (actual fill prices)
  - Postgres minute_bars (intraday replay for counterfactuals)

This script does NOT modify any bot state. Pure analytics.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "postgresql://rudra:rudra_dev_2024@localhost:5432/rudra_dev",
)


# -------------------- 1. Log parsing -----------------------------------------

@dataclass
class SignalEvent:
    timestamp: datetime
    symbol: str
    strategy: str
    action: str            # signal_buy, signal_sell, accepted, skip, ...
    reason: str
    raw_message: str
    # classifier read at same tick (if present)
    cls_long: Optional[float] = None
    cls_short: Optional[float] = None
    cls_rule_allowed: Optional[str] = None
    cls_allows_long: Optional[bool] = None
    cls_allows_short: Optional[bool] = None
    # mode at the time
    mode: Optional[str] = None
    # extra fields the strategy emitted
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Accept:
    timestamp: datetime
    symbol: str
    strategy: str
    side: str
    qty: int
    entry: float
    stop: float
    target: float
    mode: str
    # The engine's `signal_router accepted reason=<strategy>` field. Captures
    # signal.reasoning.get('strategy') directly from the engine, so it always
    # exists even when the strategy_decision join window misses (e.g. when
    # signal_buy and signal_router accepted are >5s apart). Used as fallback.
    reason: Optional[str] = None
    # paired classifier read (joined later)
    cls_long: Optional[float] = None
    cls_short: Optional[float] = None
    cls_rule_allowed: Optional[str] = None


_STRATEGY_ALIAS = {
    # The engine's `signal.reasoning['strategy']` uses long names while
    # the per-strategy `_log_decision` line uses short names. Map both to a
    # single canonical bucket so report rows aggregate cleanly.
    "free_news_sentiment": "news",
    "news_sentiment": "news",
}


def normalize_strategy(name: Optional[str]) -> str:
    """Canonicalize strategy names so news_sentiment + free_news_sentiment
    + news all roll up under 'news'. Unknown names pass through unchanged."""
    if not name or name in ("?", "unknown"):
        return name or "?"
    return _STRATEGY_ALIAS.get(name, name)


def _kv(message: str) -> Dict[str, str]:
    """Extract key=value tokens; tolerates quoted lists like rule_allowed=['long','short']."""
    out: Dict[str, str] = {}
    # First, extract bracketed list values for rule_allowed and similar.
    list_matches = re.findall(r"(\w+)=(\[[^\]]*\])", message)
    for k, v in list_matches:
        out[k] = v
    # Remove those from the string so the scalar regex doesn't get confused.
    cleaned = re.sub(r"\w+=\[[^\]]*\]", "", message)
    for k, v in re.findall(r"(\w+)=([^\s]+)", cleaned):
        if k not in out:
            out[k] = v
    return out


def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace(",", "."))
    except Exception:
        return None


def iter_log_files() -> List[Path]:
    """Return main + rotated logs, oldest first."""
    files = sorted(_REPO.glob("trading_bot.log*"), key=lambda p: p.stat().st_mtime)
    return files


def parse_logs() -> Tuple[List[SignalEvent], List[Accept]]:
    """Parse every audit line into structured events."""
    signals: List[SignalEvent] = []
    accepts: List[Accept] = []

    for path in iter_log_files():
        opener = gzip.open if path.suffix == ".gz" else open
        try:
            with opener(path, "rt") as fh:
                for line in fh:
                    if "{" not in line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    msg = d.get("message", "")
                    ts = _parse_iso(d.get("timestamp", ""))
                    if ts is None:
                        continue

                    # strategy_decision lines: any signal_buy / signal_sell /
                    # skip / advisory / bypass with strategy= and symbol=.
                    if "strategy_decision " in msg or msg.startswith("strategy_decision"):
                        kv = _kv(msg)
                        signals.append(SignalEvent(
                            timestamp=ts,
                            symbol=kv.get("symbol", "?"),
                            strategy=kv.get("strategy", "?"),
                            action=kv.get("action", "?"),
                            reason=kv.get("reason", ""),
                            raw_message=msg,
                            extra=kv,
                        ))
                        continue

                    # engine_decision: classifier shadow, signal_router actions
                    if "engine_decision " in msg or msg.startswith("engine_decision"):
                        kv = _kv(msg)
                        comp = kv.get("component", "")
                        sym = kv.get("symbol", "?")
                        action = kv.get("action", "?")
                        if comp == "side_classifier" and action == "shadow":
                            # store as a special "synthetic" SignalEvent
                            signals.append(SignalEvent(
                                timestamp=ts,
                                symbol=sym,
                                strategy="_classifier",
                                action="shadow",
                                reason=kv.get("reason", ""),
                                raw_message=msg,
                                cls_long=_floatish(kv.get("long_score")),
                                cls_short=_floatish(kv.get("short_score")),
                                cls_rule_allowed=kv.get("rule_allowed"),
                                cls_allows_long=kv.get("allows_long") == "True",
                                cls_allows_short=kv.get("allows_short") == "True",
                                mode=kv.get("mode"),
                                extra=kv,
                            ))
                        elif comp == "signal_router" and action == "accepted":
                            accepts.append(Accept(
                                timestamp=ts,
                                symbol=sym,
                                strategy="?",  # joined below
                                side=kv.get("side", "?"),
                                qty=int(float(kv.get("qty", 0) or 0)),
                                entry=float(kv.get("entry", 0) or 0),
                                stop=float(kv.get("stop", 0) or 0),
                                target=float(kv.get("target", 0) or 0),
                                mode=kv.get("mode", "?"),
                                reason=kv.get("reason"),
                            ))
        except Exception as exc:
            print(f"  [warn] failed to parse {path.name}: {exc}", file=sys.stderr)

    return signals, accepts


def _floatish(v):
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


# -------------------- 2. Schwab fill reconciliation --------------------------

_schwab_client_cache: Optional[Any] = None


def _strip_env_quotes(val: str) -> str:
    """Strip surrounding quote characters from a .env value.

    python-dotenv strips these automatically; our hand-rolled parser did
    not, which led to passing literal "quoted" credentials to Schwab and
    getting back invalid_client errors that looked like token expiry but
    were really credential corruption (2026-05-23 incident).
    """
    val = val.strip()
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
        val = val[1:-1]
    return val


def get_schwab_client():
    """Singleton authenticated Schwab client. Shared by fill-fetch and
    minute-bar replay so we don't burn two auth handshakes per run."""
    global _schwab_client_cache
    if _schwab_client_cache is not None:
        return _schwab_client_cache
    from schwab.auth import client_from_token_file
    env = (_REPO / ".env").read_text()
    api_key = _strip_env_quotes(env.split("SCHWAB_API_KEY=")[1].split("\n")[0])
    app_secret = _strip_env_quotes(env.split("SCHWAB_APP_SECRET=")[1].split("\n")[0])
    _schwab_client_cache = client_from_token_file(
        token_path=str(_REPO / "token_1.json"),
        api_key=api_key,
        app_secret=app_secret,
    )
    return _schwab_client_cache


def fetch_schwab_fills(days: int = 7) -> Dict[str, Dict[str, Any]]:
    """Fetch actual buy/sell executions from Schwab. Uses shared client."""
    try:
        client = get_schwab_client()
        hash_id = client.get_account_numbers().json()[0]["hashValue"]
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=days)
        resp = client.get_orders_for_account(
            hash_id,
            from_entered_datetime=from_date,
            to_entered_datetime=to_date,
        )
        if resp.status_code != 200:
            print(f"  [warn] schwab orders HTTP {resp.status_code}", file=sys.stderr)
            return {}
        orders = resp.json()
    except Exception as exc:
        print(f"  [warn] schwab fills fetch failed: {exc}", file=sys.stderr)
        return {}

    # Aggregate per-symbol BUY / SELL totals.
    fills_by_sym: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"buys": [], "sells": []}
    )

    def _record(sym, instr, px, q, t):
        if px <= 0 or q <= 0:
            return
        fills_by_sym[sym][instr.lower() + "s"].append({
            "px": px, "qty": q, "time": t,
        })

    for o in orders:
        for leg in o.get("orderLegCollection", []) or []:
            sym = leg.get("instrument", {}).get("symbol", "?")
            instr = leg.get("instruction", "?")
            for act in o.get("orderActivityCollection", []):
                for el in act.get("executionLegs", []):
                    _record(sym, instr,
                            float(el.get("price") or 0),
                            float(el.get("quantity") or 0),
                            el.get("time", ""))
        for child in o.get("childOrderStrategies", []) or []:
            for leg in child.get("orderLegCollection", []) or []:
                sym = leg.get("instrument", {}).get("symbol", "?")
                instr = leg.get("instruction", "?")
                for act in child.get("orderActivityCollection", []):
                    for el in act.get("executionLegs", []):
                        _record(sym, instr,
                                float(el.get("price") or 0),
                                float(el.get("quantity") or 0),
                                el.get("time", ""))

    return fills_by_sym


# -------------------- 3. Per-trade outcome computation -----------------------

@dataclass
class TradeOutcome:
    symbol: str
    entry_ts: datetime
    side: str          # BUY or SELL_SHORT
    entry_px: float
    exit_px: Optional[float]
    qty: int
    pnl: float
    hold_minutes: Optional[float]
    exit_type: str     # target / stop / manual / time / open
    strategy: Optional[str] = None
    cls_long: Optional[float] = None
    cls_short: Optional[float] = None


def reconcile_trades(
    accepts: List[Accept],
    signals: List[SignalEvent],
    fills: Dict[str, Dict[str, Any]],
) -> List[TradeOutcome]:
    """For every signal_router=accepted event, find the matching Schwab
    BUY fill and the eventual exit SELL fill. Compute realized P&L."""
    outcomes: List[TradeOutcome] = []

    # Index classifier reads + signals by (sym, second-truncated-ts) for
    # quick lookup; tolerate 2-3 second skew.
    cls_by_sym: Dict[str, List[SignalEvent]] = defaultdict(list)
    strat_by_sym: Dict[str, List[SignalEvent]] = defaultdict(list)
    for s in signals:
        if s.action == "shadow":
            cls_by_sym[s.symbol].append(s)
        elif s.action in ("signal_buy", "signal_sell"):
            strat_by_sym[s.symbol].append(s)

    for acc in accepts:
        # Find the matching strategy signal. Widened from 5s to 30s on
        # 2026-05-23 — IONQ accept showed 12s skew between signal_buy
        # and signal_router accepted (12:00:09 vs 11:59:57) due to risk
        # checks and Schwab order placement latency, which caused the
        # join to fail and the trade to show strategy="?".
        # Pick the closest strategy_decision within the window.
        strat = None
        best_dt = None
        cls = None
        for s in strat_by_sym.get(acc.symbol, []):
            dt = abs((s.timestamp - acc.timestamp).total_seconds())
            if dt < 30 and (best_dt is None or dt < best_dt):
                strat = s.strategy
                best_dt = dt
        # Fallback: if no strategy_decision matched, use the reason field
        # captured from the signal_router=accepted line itself. The engine
        # emits signal.reasoning['strategy'] there directly, so it always
        # exists when the engine accepted the signal.
        if strat is None and acc.reason:
            strat = acc.reason
        strat = normalize_strategy(strat)
        for c in cls_by_sym.get(acc.symbol, []):
            if abs((c.timestamp - acc.timestamp).total_seconds()) < 30:
                cls = c
                break

        # Find buy fill near accept timestamp; then earliest sell fill after.
        sym_fills = fills.get(acc.symbol, {"buys": [], "sells": []})
        accept_t = acc.timestamp
        # Schwab times: "2026-05-22T13:44:02+0000" (no colon in offset) OR
        # "...Z". Python 3.10 fromisoformat needs "+HH:MM" with colon; pre-normalize.
        def _parse_fill_ts(s: str) -> Optional[datetime]:
            try:
                t = s.replace("Z", "+00:00")
                m = re.match(r"(.*[+-])(\d{2})(\d{2})$", t)
                if m:
                    t = f"{m.group(1)}{m.group(2)}:{m.group(3)}"
                return datetime.fromisoformat(t)
            except Exception:
                return None

        # accept_t is naive ET; convert to UTC-aware (EDT = UTC-4 in May).
        accept_utc = (accept_t + timedelta(hours=4)).replace(tzinfo=timezone.utc) \
            if accept_t.tzinfo is None else accept_t

        # Buys within ~120s of accept.
        candidate_buys = []
        for f in sym_fills["buys"]:
            ft = _parse_fill_ts(f["time"])
            if ft is None:
                continue
            if abs((ft - accept_utc).total_seconds()) < 120:
                candidate_buys.append((ft, f))
        if not candidate_buys:
            continue
        candidate_buys.sort(key=lambda x: x[0])
        buy_ft, buy = candidate_buys[0]
        # Earliest sell AFTER this buy.
        sells_after = [
            (_parse_fill_ts(f["time"]), f) for f in sym_fills["sells"]
            if _parse_fill_ts(f["time"]) and _parse_fill_ts(f["time"]) > buy_ft
        ]
        sells_after.sort(key=lambda x: x[0])
        if not sells_after:
            # Position still open.
            outcomes.append(TradeOutcome(
                symbol=acc.symbol,
                entry_ts=acc.timestamp,
                side=acc.side,
                entry_px=buy["px"],
                exit_px=None,
                qty=acc.qty,
                pnl=0.0,
                hold_minutes=None,
                exit_type="open",
                strategy=strat,
                cls_long=cls.cls_long if cls else None,
                cls_short=cls.cls_short if cls else None,
            ))
            continue
        sell_ft, sell = sells_after[0]
        # Sum to match qty (multi-fill exits).
        remaining = acc.qty
        sell_total_px_qty = 0.0
        sell_total_qty = 0.0
        for sft, sf in sells_after:
            if remaining <= 0:
                break
            take = min(remaining, sf["qty"])
            sell_total_px_qty += sf["px"] * take
            sell_total_qty += take
            remaining -= take
        avg_sell = sell_total_px_qty / sell_total_qty if sell_total_qty else 0
        pnl = (avg_sell - buy["px"]) * acc.qty
        hold_min = (sell_ft - buy_ft).total_seconds() / 60.0
        # Exit type heuristic
        exit_type = "target" if avg_sell > acc.target * 0.997 else (
            "stop" if avg_sell < acc.stop * 1.003 else "manual"
        )
        outcomes.append(TradeOutcome(
            symbol=acc.symbol,
            entry_ts=acc.timestamp,
            side=acc.side,
            entry_px=buy["px"],
            exit_px=avg_sell,
            qty=acc.qty,
            pnl=pnl,
            hold_minutes=hold_min,
            exit_type=exit_type,
            strategy=strat,
            cls_long=cls.cls_long if cls else None,
            cls_short=cls.cls_short if cls else None,
        ))

    return outcomes


# -------------------- 4. Short-side counterfactual ---------------------------

def _fetch_schwab_minute_bars(symbol: str, start: datetime, end: datetime) -> List[Dict[str, Any]]:
    """Fetch 1-minute Schwab candles for [start, end] window.

    Schwab `get_price_history_every_minute` accepts `start_datetime` /
    `end_datetime` (Schwab API requires aware datetimes — naive ET ts
    from logs must be converted to UTC first).
    """
    client = get_schwab_client()
    # Convert naive ET to aware UTC (May = EDT = UTC-4).
    if start.tzinfo is None:
        start_utc = (start + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    else:
        start_utc = start
    if end.tzinfo is None:
        end_utc = (end + timedelta(hours=4)).replace(tzinfo=timezone.utc)
    else:
        end_utc = end
    try:
        resp = client.get_price_history_every_minute(
            symbol,
            start_datetime=start_utc,
            end_datetime=end_utc,
        )
    except Exception as exc:
        print(f"  [warn] schwab bars {symbol}: {exc}", file=sys.stderr)
        return []
    if resp.status_code != 200:
        return []
    data = resp.json()
    return data.get("candles", []) or []


async def short_counterfactual(
    short_signals: List[SignalEvent],
) -> List[TradeOutcome]:
    """For every classifier shadow with rule_allowed=['short'] AND long_score<0.30,
    simulate a short trade using authoritative Schwab 1-minute bars.

    Architectural choice: Schwab is already our primary broker; reusing the
    authenticated client here (rather than scraping yfinance) keeps the data
    pipeline single-sourced. Schwab's `get_price_history_every_minute`
    returns clean ms-epoch OHLCV — no tz-acrobatics needed inside Python.
    """
    outcomes: List[TradeOutcome] = []
    # Batch by symbol to minimize API calls — fetch each symbol's full
    # signal-window range once.
    by_sym: Dict[str, List[SignalEvent]] = defaultdict(list)
    for s in short_signals:
        by_sym[s.symbol].append(s)

    for sym, sigs in by_sym.items():
        # Union of all signal windows for this symbol.
        win_start = min(s.timestamp for s in sigs)
        win_end = max(s.timestamp for s in sigs) + timedelta(hours=4)
        candles = await asyncio.to_thread(_fetch_schwab_minute_bars, sym, win_start, win_end)
        if not candles:
            continue

        # Convert ms-epoch to naive ET timestamps for matching log ts.
        bars: List[Tuple[datetime, float, float, float, float]] = []
        for c in candles:
            try:
                ts_ms = c.get("datetime") or c.get("dateTime")
                if ts_ms is None:
                    continue
                ts_et = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc) \
                    .astimezone(timezone(timedelta(hours=-4))).replace(tzinfo=None)
                bars.append((
                    ts_et,
                    float(c["open"]),
                    float(c["high"]),
                    float(c["low"]),
                    float(c["close"]),
                ))
            except Exception:
                continue
        bars.sort(key=lambda x: x[0])
        if len(bars) < 5:
            continue

        for sig in sigs:
            # Slice from signal time to +4h.
            start = sig.timestamp
            end = start + timedelta(hours=4)
            window = [b for b in bars if start <= b[0] <= end]
            if len(window) < 5:
                continue
            entry_px = window[0][4]    # close of first bar
            stop_px = entry_px * 1.02
            target_px = entry_px * 0.96
            exit_px = None
            exit_type = "time"
            exit_ts = None
            for ts_b, _o, hi, lo, _cl in window[1:]:
                if hi >= stop_px:
                    exit_px, exit_type, exit_ts = stop_px, "stop", ts_b
                    break
                if lo <= target_px:
                    exit_px, exit_type, exit_ts = target_px, "target", ts_b
                    break
            if exit_px is None:
                exit_px = window[-1][4]
                exit_ts = window[-1][0]
            stop_dist = stop_px - entry_px
            qty = max(1, int(200 / max(stop_dist, 0.01)))
            pnl = (entry_px - exit_px) * qty
            hold_min = ((exit_ts - start).total_seconds() / 60.0) if exit_ts else None
            outcomes.append(TradeOutcome(
                symbol=sig.symbol,
                entry_ts=sig.timestamp,
                side="SELL_SHORT_SIM",
                entry_px=entry_px,
                exit_px=exit_px,
                qty=qty,
                pnl=pnl,
                hold_minutes=hold_min,
                exit_type=exit_type,
                strategy="_classifier_short_signal",
                cls_long=sig.cls_long,
                cls_short=sig.cls_short,
            ))

    return outcomes


# -------------------- 5. Slice analysis --------------------------------------

def classify_symbol(sym: str) -> str:
    leveraged = {
        "TQQQ","SQQQ","UPRO","SPXU","SPXL","SPXS","TNA","TZA","FAS","FAZ",
        "SOXL","SOXS","LABU","LABD","NUGT","DUST","JNUG","JDST","TMF","TMV",
        "UVXY","SVXY","VXX","VIXY","TSLL","TSLZ","NVDL","NVDS","MSTU","MSTX",
        "USO","UCO","SCO","AGQ","ZSL","UGL","GLL","TECL","TECS","ERX","ERY",
        "URTY","SRTY","QLD","QID","SSO","SDS",
    }
    if sym in leveraged:
        return "leveraged_etf"
    mega = {"AAPL","MSFT","GOOG","GOOGL","NVDA","AMD","META","AMZN","TSLA","AVGO","NFLX"}
    if sym in mega:
        return "mega_cap"
    mid = {"PLTR","COIN","RKLB","INTC","MU","IBM","ORCL","CRM","ADBE","NOW","PYPL","HOOD"}
    if sym in mid:
        return "mid_cap"
    return "small_or_other"


def classify_time_slot(ts: datetime) -> str:
    """Bucket by intraday time. Times are ET / log-local."""
    t = ts.time()
    if time(9, 30) <= t < time(9, 35):
        return "first_5min"
    if time(9, 35) <= t < time(10, 30):
        return "morning_active"
    if time(10, 30) <= t < time(12, 0):
        return "late_morning"
    if time(12, 0) <= t < time(15, 0):
        return "midday"
    if time(15, 0) <= t < time(16, 0):
        return "close_hour"
    return "outside_rth"


def classify_classifier_bucket(long_score: Optional[float]) -> str:
    if long_score is None:
        return "no_data"
    if long_score < 0.25:
        return "strong_bearish_<0.25"
    if long_score < 0.35:
        return "bearish_0.25-0.35"
    if long_score < 0.40:
        return "neutral_0.35-0.40"
    return "agree_>=0.40"


def summarize_slice(name: str, trades: List[TradeOutcome]) -> Dict[str, Any]:
    closed = [t for t in trades if t.exit_px is not None]
    if not closed:
        return {"name": name, "n": 0}
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    total_pnl = sum(t.pnl for t in closed)
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0
    return {
        "name": name,
        "n": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "avg_pnl": round(total_pnl / len(closed), 2),
        "avg_win": round(gross_win / len(wins), 2) if wins else 0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float('inf'),
        "avg_hold_min": round(sum((t.hold_minutes or 0) for t in closed) / len(closed), 1),
    }


# -------------------- 6. Markdown report writer ------------------------------

def write_report(
    signals: List[SignalEvent],
    accepts: List[Accept],
    long_outcomes: List[TradeOutcome],
    short_counterfactuals: List[TradeOutcome],
    out_path: Path,
):
    lines: List[str] = []
    lines.append(f"# Weekly Trading Bot Report — generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("**Stop patching. Measure honestly. Decide based on data.**")
    lines.append("")

    # --- Section 1: headline numbers
    closed = [t for t in long_outcomes if t.exit_px is not None]
    realized = sum(t.pnl for t in closed)
    wins = sum(1 for t in closed if t.pnl > 0)
    lines.append("## Section 1 — Headline")
    lines.append("")
    lines.append(f"- Logs parsed: {sum(1 for _ in iter_log_files())} files")
    lines.append(f"- Strategy signals (any action): **{len(signals)}**")
    lines.append(f"- Accepted entries (signal_router): **{len(accepts)}**")
    lines.append(f"- Closed trades (matched to Schwab fills): **{len(closed)}**")
    lines.append(f"- **Realized P&L (closed long trades): ${realized:+,.2f}**")
    lines.append(f"- Win rate: **{wins}/{len(closed)} = {wins/max(len(closed),1)*100:.1f}%**")
    lines.append("")

    # --- Section 2: by strategy
    lines.append("## Section 2 — Per-strategy P&L")
    lines.append("")
    by_strat: Dict[str, List[TradeOutcome]] = defaultdict(list)
    for t in long_outcomes:
        by_strat[t.strategy or "unknown"].append(t)
    lines.append("| Strategy | N | WR | PF | Net P&L | Avg Win | Avg Loss | Avg Hold (min) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in sorted(by_strat.keys()):
        d = summarize_slice(s, by_strat[s])
        if d.get('n', 0) == 0:
            continue
        lines.append(
            f"| {s} | {d['n']} | {d['win_rate_pct']}% | {d['profit_factor']} | "
            f"${d['total_pnl']:+,.2f} | ${d['avg_win']:+,.2f} | ${d['avg_loss']:+,.2f} | "
            f"{d['avg_hold_min']} |"
        )
    lines.append("")

    # --- Section 3: by symbol class
    lines.append("## Section 3 — Per-symbol-class P&L")
    lines.append("")
    by_class: Dict[str, List[TradeOutcome]] = defaultdict(list)
    for t in long_outcomes:
        by_class[classify_symbol(t.symbol)].append(t)
    lines.append("| Class | N | WR | PF | Net P&L |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in sorted(by_class.keys()):
        d = summarize_slice(c, by_class[c])
        if d.get('n', 0) == 0:
            continue
        lines.append(f"| {c} | {d['n']} | {d['win_rate_pct']}% | {d['profit_factor']} | ${d['total_pnl']:+,.2f} |")
    lines.append("")

    # --- Section 4: by time-of-day
    lines.append("## Section 4 — Per-time-slot P&L")
    lines.append("")
    by_slot: Dict[str, List[TradeOutcome]] = defaultdict(list)
    for t in long_outcomes:
        by_slot[classify_time_slot(t.entry_ts)].append(t)
    lines.append("| Slot | N | WR | PF | Net P&L | Avg Hold |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in ["first_5min", "morning_active", "late_morning", "midday", "close_hour", "outside_rth"]:
        d = summarize_slice(s, by_slot.get(s, []))
        if d.get('n', 0) == 0:
            continue
        lines.append(f"| {s} | {d['n']} | {d['win_rate_pct']}% | {d['profit_factor']} | ${d['total_pnl']:+,.2f} | {d['avg_hold_min']}m |")
    lines.append("")

    # --- Section 5: by classifier bucket
    lines.append("## Section 5 — Per-classifier-reading P&L")
    lines.append("")
    by_cls: Dict[str, List[TradeOutcome]] = defaultdict(list)
    for t in long_outcomes:
        by_cls[classify_classifier_bucket(t.cls_long)].append(t)
    lines.append("| Classifier bucket | N | WR | PF | Net P&L |")
    lines.append("|---|---:|---:|---:|---:|")
    for c in ["strong_bearish_<0.25", "bearish_0.25-0.35", "neutral_0.35-0.40", "agree_>=0.40", "no_data"]:
        d = summarize_slice(c, by_cls.get(c, []))
        if d.get('n', 0) == 0:
            continue
        lines.append(f"| {c} | {d['n']} | {d['win_rate_pct']}% | {d['profit_factor']} | ${d['total_pnl']:+,.2f} |")
    lines.append("")

    # --- Section 6: Short counterfactual
    lines.append("## Section 6 — Short-side counterfactual")
    lines.append("")
    lines.append("For every classifier shadow signal with `rule_allowed=['short']` AND `long_score < 0.30`,")
    lines.append("simulated a 2%/4% R:R short trade using Schwab 1-min bars (same broker as live trading).")
    lines.append("**This is the alpha we left on the table by being long-only.**")
    lines.append("")
    if not short_counterfactuals:
        lines.append("> No short counterfactual data available (Postgres bars missing or no signals matched).")
    else:
        sd = summarize_slice("short_counterfactual", short_counterfactuals)
        lines.append(f"- Simulated short trades: **{sd['n']}**")
        lines.append(f"- Hypothetical realized: **${sd['total_pnl']:+,.2f}**")
        lines.append(f"- Win rate: **{sd['win_rate_pct']}%**")
        lines.append(f"- Profit factor: **{sd['profit_factor']}**")
        lines.append(f"- Avg hold: {sd['avg_hold_min']} min")
        lines.append("")
        lines.append("### Per-strategy-of-original-signal")
        # Group by strategy of the original signal
        by_orig: Dict[str, List[TradeOutcome]] = defaultdict(list)
        for t in short_counterfactuals:
            by_orig[t.strategy or "unknown"].append(t)
        lines.append("| Source signal | N | WR | Net P&L |")
        lines.append("|---|---:|---:|---:|")
        for s, tt in by_orig.items():
            d = summarize_slice(s, tt)
            if d.get('n', 0) == 0:
                continue
            lines.append(f"| {s} | {d['n']} | {d['win_rate_pct']}% | ${d['total_pnl']:+,.2f} |")
    lines.append("")

    # --- Section 7: trade list (for audit)
    lines.append("## Section 7 — Closed trade list")
    lines.append("")
    lines.append("| Sym | Strategy | Entry | Exit | Side | PnL | Hold (min) | Exit | cls_long |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|---:|")
    for t in sorted(closed, key=lambda x: x.entry_ts):
        cls_disp = f"{t.cls_long:.3f}" if t.cls_long is not None else "—"
        hold_disp = f"{t.hold_minutes:.1f}" if t.hold_minutes is not None else "—"
        lines.append(
            f"| {t.symbol} | {t.strategy or '?'} | ${t.entry_px:.2f} | ${t.exit_px:.2f} | "
            f"{t.side} | ${t.pnl:+,.2f} | {hold_disp} | "
            f"{t.exit_type} | {cls_disp} |"
        )
    lines.append("")

    # --- Section 8: Decisions implied by the data
    lines.append("## Section 8 — Decisions implied by the data")
    lines.append("")
    lines.append("(To be filled in collaboratively after review — see action checklist below.)")
    lines.append("")
    lines.append("- [ ] Which strategy has positive PF and adequate sample? → KEEP")
    lines.append("- [ ] Which strategy has PF < 1.0 or n < 10? → DISABLE")
    lines.append("- [ ] Which time slot has worst PF? → ADD TIME GATE")
    lines.append("- [ ] Which classifier bucket should gate trades? → PROMOTE TO LIVE GATE")
    lines.append("- [ ] Is short counterfactual edge > $0? → BUILD SHORT BRANCH")
    lines.append("- [ ] Which symbol class to allowlist / blocklist?")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# -------------------- 7. Main ------------------------------------------------

async def main():
    print("[1/5] parsing logs (this + rotated)...")
    signals, accepts = parse_logs()
    print(f"      → {len(signals)} signal events, {len(accepts)} accepts")

    print("[2/5] fetching Schwab fills (7 days)...")
    fills = fetch_schwab_fills(days=7)
    n_fills = sum(len(v.get("buys", [])) + len(v.get("sells", [])) for v in fills.values())
    print(f"      → {n_fills} fill records across {len(fills)} symbols")

    print("[3/5] reconciling trades to outcomes...")
    long_outcomes = reconcile_trades(accepts, signals, fills)
    print(f"      → {len(long_outcomes)} reconciled trades")

    print("[4/5] short counterfactual sim (Schwab 1-min bars)...")
    # Pick the strong-bearish classifier shadow signals.
    short_signals = [
        s for s in signals
        if s.action == "shadow"
        and s.cls_rule_allowed and "short" in s.cls_rule_allowed
        and (s.cls_long is not None and s.cls_long < 0.30)
    ]
    print(f"      → {len(short_signals)} qualifying short signals")
    short_outcomes = await short_counterfactual(short_signals)
    print(f"      → {len(short_outcomes)} short counterfactual outcomes")

    print("[5/5] writing report...")
    out = _REPO / "docs" / f"weekly_report_{date.today().isoformat()}.md"
    write_report(signals, accepts, long_outcomes, short_outcomes, out)
    print(f"      → {out}")
    print()
    print("Done. Read the report, then we discuss what to keep / cut.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
