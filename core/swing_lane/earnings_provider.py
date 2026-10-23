"""v-swing-lane-2026-10-23: earnings-calendar provider for the swing lane.

Flag SWING_EARNINGS_PROVIDER (env or yaml trading.swing_earnings_provider):
  none    (default) legacy: read <root>/earnings/<SYM>.parquet only
  nasdaq  Nasdaq public calendar (api.nasdaq.com/api/calendar/earnings?date=), one call/day in window
  yahoo   Yahoo quoteSummary calendarEvents (cookie+crumb, per symbol)
  auto    Nasdaq primary, Yahoo fallback; when both succeed their dates are UNIONED (more blackout, never less)

Cache: <log_dir>/earnings_cache.json, refreshed at most once per ET day (refresh()).
Fail closed: dates(sym) returns None (=> earnings_data_missing) when the cache is absent,
stale (> MAX_AGE_DAYS), or the symbol is not covered by any successful source.
An empty set means "covered and no report in the window".
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time as _time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger("swing_lane")

PROVIDERS = ("none", "nasdaq", "yahoo", "auto")
CACHE_NAME = "earnings_cache.json"
WINDOW_BACK_DAYS = 10     # calendar days before today (setup C: report in prior trading day)
WINDOW_FWD_DAYS = 28      # calendar days ahead (5-day blackout + 10-day hold)
MAX_AGE_DAYS = 3          # cache older than this => unknown (covers a weekend + one failed refresh)
NASDAQ_URL = "https://api.nasdaq.com/api/calendar/earnings?date={d}"
YAHOO_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{s}?modules=calendarEvents&crumb={c}"
YAHOO_COOKIE_URL = "https://fc.yahoo.com"
YAHOO_CRUMB_URL = "https://query1.finance.yahoo.com/v1/test/getcrumb"
_NASDAQ_TIME = {"time-pre-market": "bmo", "time-after-hours": "amc"}


def normalize_provider(v) -> str:
    v = (str(v or "none")).strip().lower()
    return v if v in PROVIDERS else "none"


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


# ── source parsers (pure; fixture-tested) ────────────────────────────────
def parse_nasdaq_day(payload: dict) -> Optional[dict]:
    """-> {SYM: timing} for one day, {} for a valid empty day, None if malformed/error."""
    if not isinstance(payload, dict):
        return None
    st = payload.get("status") or {}
    if st.get("rCode") != 200 or not isinstance(payload.get("data"), dict):
        return None
    rows = payload["data"].get("rows") or []
    out = {}
    for r in rows:
        sym = str(r.get("symbol") or "").strip().upper()
        if sym:
            out[sym] = _NASDAQ_TIME.get(r.get("time"), "unknown")
    return out


def parse_yahoo(payload: dict) -> Optional[list]:
    """-> list of (date_iso, is_estimate) or None when the response is unusable."""
    try:
        res = payload["quoteSummary"]["result"]
        if not res:
            return None
        e = (res[0].get("calendarEvents") or {}).get("earnings") or {}
    except (KeyError, TypeError, IndexError):
        return None
    raw = e.get("earningsDate")
    if raw is None:
        return None
    est = bool(e.get("isEarningsDateEstimate", False))
    out = []
    for x in raw:
        if not isinstance(x, dict):
            continue
        fmt = x.get("fmt")
        if isinstance(fmt, str) and len(fmt) == 10:
            out.append((date.fromisoformat(fmt).isoformat(), est))  # Yahoo's own calendar day
        elif x.get("raw") is not None:
            out.append((datetime.fromtimestamp(int(x["raw"]), timezone.utc).date().isoformat(), est))
    return out


# ── provider ─────────────────────────────────────────────────────────────
class EarningsProvider:
    def __init__(self, provider: str, cache_dir: str, symbols: Iterable[str] = (),
                 fetch_json: Optional[Callable[[str], dict]] = None,
                 fetch_text: Optional[Callable[[str], str]] = None,
                 sleep: Callable[[float], None] = _time.sleep, pause_s: float = 0.4):
        self.provider = normalize_provider(provider)
        self.path = Path(cache_dir) / CACHE_NAME
        self.symbols = sorted({s.upper() for s in symbols})
        self._fetch_json = fetch_json
        self._fetch_text = fetch_text
        self._sleep = sleep
        self.pause_s = pause_s
        self._cache = None

    # transport (lazy so tests/flag-off never touch the network module)
    def _http(self):
        if self._fetch_json is None or self._fetch_text is None:
            from core.swing_lane.earnings_http import Http
            h = Http()
            self._fetch_json = self._fetch_json or h.get_json
            self._fetch_text = self._fetch_text or h.get_text
        return self._fetch_json, self._fetch_text

    # cache io
    def load(self) -> Optional[dict]:
        if self._cache is None and self.path.exists():
            try:
                self._cache = json.loads(self.path.read_text())
            except Exception:
                logger.warning("swing_lane: unreadable earnings cache %s", self.path)
                self._cache = None
        return self._cache

    def _save(self, cache: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".earn", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(cache, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)
        self._cache = cache

    def needs_refresh(self, today: date) -> bool:
        c = self.load()
        return self.provider != "none" and (not c or c.get("fetched_on") != today.isoformat()
                                            or c.get("provider") != self.provider)

    # sources
    def _fetch_nasdaq(self, today: date) -> Optional[dict]:
        """All weekdays in the window must succeed, else None (partial coverage is not coverage)."""
        fj, _ = self._http()
        dates: dict = {}
        timing: dict = {}
        for d in _weekdays(today - timedelta(days=WINDOW_BACK_DAYS), today + timedelta(days=WINDOW_FWD_DAYS)):
            day = None
            for attempt in range(2):
                try:
                    day = parse_nasdaq_day(fj(NASDAQ_URL.format(d=d.isoformat())))
                except Exception as exc:
                    logger.info("swing_lane: nasdaq %s failed: %s", d, exc)
                    day = None
                if day is not None:
                    break
                self._sleep(2.0)
            if day is None:
                logger.warning("swing_lane: nasdaq calendar incomplete at %s; source unusable today", d)
                return None
            for sym, t in day.items():
                dates.setdefault(sym, []).append(d.isoformat())
                timing[f"{sym}|{d.isoformat()}"] = t
            self._sleep(self.pause_s)
        return {"dates": dates, "timing": timing, "all_symbols": True}

    def _fetch_yahoo(self) -> Optional[dict]:
        fj, ft = self._http()
        try:
            ft(YAHOO_COOKIE_URL)
            crumb = (ft(YAHOO_CRUMB_URL) or "").strip()
        except Exception as exc:
            logger.warning("swing_lane: yahoo crumb failed: %s", exc)
            return None
        if not crumb or "<" in crumb or len(crumb) > 64:
            return None
        dates, covered, est = {}, [], {}
        for s in self.symbols:
            try:
                got = parse_yahoo(fj(YAHOO_URL.format(s=s, c=crumb)))
            except Exception as exc:
                logger.info("swing_lane: yahoo %s failed: %s", s, exc)
                got = None
            if got is not None:
                covered.append(s)
                dates[s] = [d for d, _ in got]
                if any(e for _, e in got):
                    est[s] = True
            self._sleep(self.pause_s)
        if not covered:
            return None
        return {"dates": dates, "covered": covered, "estimate": est, "all_symbols": False}

    def refresh(self, today: date, force: bool = False) -> bool:
        """Fetch per the flag and rewrite the cache. -> True if a fresh cache was written.
        On total failure the previous cache is kept (and ages out after MAX_AGE_DAYS)."""
        if self.provider == "none" or (not force and not self.needs_refresh(today)):
            return False
        nq = self._fetch_nasdaq(today) if self.provider in ("nasdaq", "auto") else None
        yh = self._fetch_yahoo() if (self.provider == "yahoo" or self.provider == "auto") else None
        if nq is None and yh is None:
            logger.warning("swing_lane: earnings refresh failed for provider=%s; entries stay fail-closed", self.provider)
            return False
        cache = {"provider": self.provider, "fetched_on": today.isoformat(),
                 "fetched_at": datetime.now().isoformat(timespec="seconds"),
                 "window": [(today - timedelta(days=WINDOW_BACK_DAYS)).isoformat(),
                            (today + timedelta(days=WINDOW_FWD_DAYS)).isoformat()],
                 "sources": {"nasdaq": nq is not None, "yahoo": yh is not None},
                 "all_symbols": bool(nq), "covered": sorted(yh["covered"]) if yh else [],
                 "dates": {}, "timing": (nq or {}).get("timing", {}),
                 "yahoo_estimate": (yh or {}).get("estimate", {})}
        for src in (nq, yh):
            for s, ds in (src or {}).get("dates", {}).items():
                cache["dates"][s] = sorted(set(cache["dates"].get(s, [])) | set(ds))
        self._save(cache)
        return True

    def dates(self, symbol: str, today: Optional[date] = None) -> Optional[set]:
        """Known earnings dates for `symbol`, or None (unknown => fail closed)."""
        c = self.load()
        if not c or c.get("provider") != self.provider:
            return None
        today = today or date.today()
        try:
            age = (today - date.fromisoformat(c["fetched_on"])).days
        except Exception:
            return None
        if age < 0 or age > MAX_AGE_DAYS:
            return None
        s = symbol.upper()
        if not (c.get("all_symbols") or s in set(c.get("covered") or ())):
            return None
        return {date.fromisoformat(d) for d in c.get("dates", {}).get(s, [])}


def make_loader(settings, universe: Iterable[str] = (), today: Optional[date] = None) -> Optional[Callable]:
    """earnings_loader for scheduler, or None to keep the legacy parquet loader."""
    prov = normalize_provider(getattr(settings, "earnings_provider", "none"))
    if prov == "none":
        return None
    ep = EarningsProvider(prov, settings.log_dir, universe)
    return lambda sym: ep.dates(sym, today)
