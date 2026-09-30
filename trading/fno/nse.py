"""Minimal client for NSE's public JSON endpoints (the only free source of
stock-futures open interest).

The site refuses API calls that arrive without the cookies its HTML pages set,
so the session is bootstrapped against the homepage before the first call and
re-bootstrapped once on a 401/403. Endpoints are undocumented and their field
spellings drift between deployments, so every parser here reads through a list
of candidate keys and returns None rather than guessing.

Nothing in this module invents a value: a field that cannot be found comes back
as None and the scanner degrades the affected candidate.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import requests

from trading.fno.models import IST

BASE = "https://www.nseindia.com"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/market-data/live-equity-market",
    "Connection": "keep-alive",
}
TIMEOUT = 12


class NSEError(RuntimeError):
    pass


class NSEClient:
    """Cookie-bootstrapped session over nseindia.com."""

    def __init__(self, timeout: int = TIMEOUT):
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update(HEADERS)
        self._ready = False

    def _bootstrap(self) -> None:
        for path in ("/", "/market-data/live-equity-market"):
            try:
                self.s.get(BASE + path, timeout=self.timeout)
            except requests.RequestException:
                pass
        self._ready = True

    def get(self, path: str, params: dict | None = None, retries: int = 2) -> Any:
        if not self._ready:
            self._bootstrap()
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                r = self.s.get(BASE + path, params=params, timeout=self.timeout)
                if r.status_code in (401, 403, 429):
                    self._ready = False
                    self._bootstrap()
                    last = NSEError(f"{r.status_code} on {path}")
                    time.sleep(1.0 + attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except (requests.RequestException, ValueError) as exc:
                last = exc
                time.sleep(0.6 * (attempt + 1))
        raise NSEError(f"NSE request failed: {path} ({last})")

    # --- endpoints ------------------------------------------------------

    # Endpoint choices below were made by probing the live site (Aug 2026).
    # /api/equity-stockIndices and /api/quote-derivative both answer 404 to
    # programmatic clients now, so nothing depends on them; the four endpoints
    # used here each returned full data on probe.

    def movers(self) -> tuple[list[dict], list[dict], str | None]:
        """F&O gainers and losers with % change, volume and traded value.

        The F&O block ("FOSec") of the variations endpoint — top 20 each way,
        which is exactly the pool an opening-window long scanner cares about.
        """
        out = []
        stamp = None
        for index in ("gainers", "loosers"):
            data = self.get("/api/live-analysis-variations", {"index": index})
            stamp = stamp or data.get("timestamp")
            out.append((data.get("FOSec") or {}).get("data", []) or [])
        return out[0], out[1], stamp

    def oi_spurts(self) -> tuple[list[dict], str | None]:
        """Latest vs previous-day open interest for every F&O underlying."""
        data = self.get("/api/live-analysis-oi-spurts-underlyings")
        return data.get("data", []) or [], data.get("timestamp")

    def most_active_underlying(self) -> list[dict]:
        """Futures volume and turnover per underlying — the liquidity screen."""
        data = self.get("/api/live-analysis-most-active-underlying")
        return data.get("data", []) or []

    def active_futures(self) -> tuple[list[dict], str | None]:
        """Most-active stock-futures contracts — the only source of the actual
        contract identifier and futures (not cash) last price."""
        data = self.get("/api/liveEquity-derivatives", {"index": "stock_fut"})
        return data.get("data", []) or [], data.get("timestamp")

    def active_options(self) -> tuple[list[dict], str | None]:
        """The 20 most-active stock option contracts.

        This is the only option data NSE serves programmatically — the
        option-chain endpoints answer an empty object — so there is no chain
        per stock, only whatever is busiest market-wide right now.
        """
        data = self.get("/api/liveEquity-derivatives", {"index": "stock_opt"})
        return data.get("data", []) or [], data.get("timestamp")

    def index_options(self, index: str = "nse50_opt") -> tuple[list[dict], str | None]:
        """The full NIFTY or BANK NIFTY option chain — every strike and expiry."""
        data = self.get("/api/liveEquity-derivatives", {"index": index})
        return data.get("data", []) or [], data.get("timestamp")

    def all_indices(self) -> tuple[list[dict], str | None]:
        """Every NSE index — NIFTY 50, BANK NIFTY and the sector indices."""
        data = self.get("/api/allIndices")
        return data.get("data", []) or [], data.get("timestamp")

    def fno_symbols(self) -> list[str]:
        """The F&O-eligible symbol list."""
        data = self.get("/api/master-quote")
        return [s for s in data if isinstance(s, str)] if isinstance(data, list) else []


# --- tolerant field access -------------------------------------------------

def pick(d: dict, *keys, default=None):
    """First present, non-empty key from `keys` (case-insensitive)."""
    if not isinstance(d, dict):
        return default
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        v = lowered.get(k.lower())
        if v not in (None, "", "-"):
            return v
    return default


def num(value) -> float | None:
    """Parse NSE's numbers, which arrive as floats, ints or '1,23,456.70'."""
    if value in (None, "", "-"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").replace("₹", "").strip())
    except ValueError:
        return None


def parse_timestamp(text: str | None) -> datetime | None:
    """NSE stamps look like '19-Aug-2026 09:47:31' or '19-Aug-2026 09:47'."""
    if not text:
        return None
    for fmt in ("%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d-%b-%Y"):
        try:
            return datetime.strptime(text.strip(), fmt).replace(tzinfo=IST)
        except ValueError:
            continue
    return None
