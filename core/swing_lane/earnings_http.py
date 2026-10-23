"""v-swing-lane-2026-10-23: the ONLY network code in core.swing_lane.

Plain HTTPS GETs to an allowlist of public earnings-calendar hosts (no broker,
no account, no keys). Anything else raises. Tests never call this; they inject
recorded fixtures through EarningsProvider(fetch_json=...).
"""
from __future__ import annotations

import json
import urllib.request
from http.cookiejar import CookieJar
from urllib.parse import urlparse

ALLOWED_HOSTS = frozenset({"api.nasdaq.com", "fc.yahoo.com",
                           "query1.finance.yahoo.com", "query2.finance.yahoo.com"})
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")


class HostNotAllowed(ValueError):
    pass


class Http:
    """Tiny cookie-keeping GET client (Yahoo needs a cookie + crumb)."""

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    def get_text(self, url: str) -> str:
        host = urlparse(url).hostname or ""
        if urlparse(url).scheme != "https" or host not in ALLOWED_HOSTS:
            raise HostNotAllowed(url)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json, text/plain, */*"})
        try:
            with self._opener.open(req, timeout=self.timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if url.startswith("https://fc.yahoo.com"):
                return ""  # fc.yahoo.com 404s but still sets the cookie
            raise

    def get_json(self, url: str):
        return json.loads(self.get_text(url))
