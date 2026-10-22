"""v-learning-loop-2026-10-22: one normalized view of closed live trades.

Rows come from Postgres (`bot_trades` + the `signal_router/accepted` audit row
for the signal's own entry/stop/target) or from a CSV export of the same query
(`LIVE_TRADES_SQL`). All reads are SELECT-only.

Exclusions (always applied): mode != live, source != bot, is_external,
is_hands_off, and any symbol in the hands-off denylist (HQGE/SPCX/... from
trading.hands_off_denylist) — those positions are never the bot's to learn from.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from core.learning.scorecard import LearnTrade

DAY_TRADE_STRATEGIES = ("day_trade_momentum", "day_trade_momentum_short")

LIVE_TRADES_SQL = """
SELECT t.id, t.symbol, t.side, t.strategy, t.setup_type,
       (t.entry_time AT TIME ZONE 'America/New_York') AS entry_et,
       (t.exit_time  AT TIME ZONE 'America/New_York') AS exit_et,
       t.entry_price, t.exit_price, t.quantity, t.pnl, t.pnl_r, t.stop_loss, t.take_profit,
       t.initial_stop, t.initial_tp, t.initial_risk_per_share, t.atr_at_entry,
       t.exit_reason, t.exit_reason_raw, t.mode, t.source, t.is_external, t.is_hands_off,
       (d.details_json->>'entry') AS sig_entry, (d.details_json->>'stop') AS sig_stop,
       (d.details_json->>'target') AS sig_target
FROM bot_trades t
LEFT JOIN LATERAL (
  SELECT details_json FROM bot_decisions d
  WHERE d.symbol = t.symbol AND d.component = 'signal_router' AND d.action = 'accepted'
    AND d.ts BETWEEN (t.entry_time AT TIME ZONE 'America/New_York') - INTERVAL '2 min'
                 AND (t.entry_time AT TIME ZONE 'America/New_York') + INTERVAL '10 s'
  ORDER BY d.ts DESC LIMIT 1) d ON TRUE
WHERE t.mode = 'live' AND t.exit_time >= :since AND t.exit_time < :until
ORDER BY t.entry_time
"""

# minute_bars.ts is naive ET (04:00-19:59 sessions).
BARS_SQL = """
SELECT ts, open AS o, high AS h, low AS l, close AS c FROM minute_bars
WHERE symbol = :symbol AND ts >= :start AND ts < :end ORDER BY ts
"""


def _f(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _b(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("t", "true", "1", "yes")
    return bool(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else False


def _naive_et(v: Any) -> Optional[pd.Timestamp]:
    ts = pd.to_datetime(v, errors="coerce", utc=False)
    if ts is None or pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("America/New_York").tz_localize(None)
    return ts


@dataclass
class Trade:
    id: Any
    symbol: str
    side: str
    strategy: str
    setup_type: str
    entry_ts: pd.Timestamp
    exit_ts: pd.Timestamp
    entry: float
    exit: float
    qty: float
    pnl: float
    stop: Optional[float]
    target: Optional[float]
    stop_source: str
    rps: Optional[float]
    r: Optional[float]
    r_source: str
    exit_reason: str
    sig_entry: Optional[float]
    fees: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def session(self) -> str:
        return self.entry_ts.strftime("%Y-%m-%d")

    @property
    def sign(self) -> int:
        return 1 if self.side == "long" else -1


def normalize(rows: Iterable[Dict[str, Any]], denylist: Iterable[str] = ()) -> List[Trade]:
    deny = {str(s).upper() for s in (denylist or ())}
    out: List[Trade] = []
    for r in rows:
        if str(r.get("mode", "live")) != "live" or str(r.get("source") or "bot") != "bot":
            continue
        if _b(r.get("is_external")) or _b(r.get("is_hands_off")):
            continue
        sym = str(r.get("symbol") or "").upper()
        if not sym or sym in deny:
            continue
        side = str(r.get("side") or "").lower()
        side = {"buy": "long", "sell": "short"}.get(side, side)
        entry, qty, pnl = _f(r.get("entry_price")), _f(r.get("quantity")), _f(r.get("pnl"))
        en, ex = _naive_et(r.get("entry_et")), _naive_et(r.get("exit_et"))
        if side not in ("long", "short") or entry is None or not qty or pnl is None or en is None or ex is None:
            continue
        stop, src = None, "none"
        for col in ("initial_stop", "sig_stop"):
            v = _f(r.get(col))
            if v and v > 0:
                stop, src = v, col
                break
        target = next((v for v in (_f(r.get("initial_tp")), _f(r.get("sig_target")), _f(r.get("take_profit")))
                       if v and v > 0), None)
        rps = _f(r.get("initial_risk_per_share"))
        if not (rps and rps > 0):
            rps = abs(entry - stop) if stop else None
            rps = rps if rps and rps > 0 else None
        rr, rsrc = _f(r.get("pnl_r")), "pnl_r"
        if rr is None and rps:
            rr, rsrc = pnl / (rps * qty), f"derived:{src}"
        if rr is None:
            rsrc = "none"
        out.append(Trade(
            id=r.get("id"), symbol=sym, side=side, strategy=str(r.get("strategy") or ""),
            setup_type=str(r.get("setup_type") or "") if r.get("setup_type") is not None and str(r.get("setup_type")) != "nan" else "",
            entry_ts=en, exit_ts=ex, entry=entry, exit=_f(r.get("exit_price")) or entry, qty=qty, pnl=pnl,
            stop=stop, target=target, stop_source=src, rps=rps, r=rr, r_source=rsrc,
            exit_reason=str(r.get("exit_reason_raw") or r.get("exit_reason") or ""),
            sig_entry=_f(r.get("sig_entry")), fees=_f(r.get("fees_usd", r.get("fees")))))
    return out


def load_csv(path: Any, denylist: Iterable[str] = ()) -> List[Trade]:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return normalize(df.to_dict("records"), denylist)


def load_bars_csv(path: Any) -> Dict[Any, pd.DataFrame]:
    """`id,ts,o,h,l,c` export → {trade_id: bars}."""
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"])
    return {str(k): g.drop(columns=["id"]).reset_index(drop=True) for k, g in df.groupby(df["id"].astype(str))}


def load_db(reader: Callable[..., pd.DataFrame], since: datetime, until: datetime,
            denylist: Iterable[str] = ()) -> List[Trade]:
    df = reader(LIVE_TRADES_SQL, {"since": since, "until": until})
    return normalize(df.to_dict("records") if df is not None else [], denylist)


def bars_for_db(reader: Callable[..., pd.DataFrame], trades: Sequence[Trade]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for t in trades:
        start = t.entry_ts.floor("min")
        end = start.normalize() + pd.Timedelta(hours=16)
        df = reader(BARS_SQL, {"symbol": t.symbol, "start": start.to_pydatetime(), "end": end.to_pydatetime()})
        if df is not None and not df.empty:
            df["ts"] = pd.to_datetime(df["ts"])
            out[str(t.id)] = df.reset_index(drop=True)
    return out


def to_learn_trades(trades: Sequence[Trade], family_for: Callable[[str], Optional[str]],
                    regime_for: Callable[[Trade], str] = lambda t: "unknown") -> List[LearnTrade]:
    out: List[LearnTrade] = []
    for t in trades:
        fam = family_for(t.strategy)
        if fam is None or t.r is None:
            continue
        out.append(LearnTrade(family=fam, side=t.side, regime=regime_for(t), r=float(t.r),
                              session=t.exit_ts.strftime("%Y-%m-%d"), exit_ts=t.exit_ts.isoformat(),
                              exit_reason=t.exit_reason, symbol=t.symbol, pnl=t.pnl, source="live"))
    return out


def sql_reader(dsn: str) -> Callable[..., pd.DataFrame]:
    """Read-only reader: every connection runs `SET TRANSACTION READ ONLY`."""
    from sqlalchemy import create_engine, text
    eng = create_engine(dsn.replace("+asyncpg", ""), pool_pre_ping=True)

    def read(q: str, params: Dict[str, Any]) -> pd.DataFrame:
        with eng.connect() as conn:
            conn.execute(text("SET TRANSACTION READ ONLY"))
            return pd.read_sql(text(q), conn, params=params)
    return read
