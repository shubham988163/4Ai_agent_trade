"""One place that runs a scan.

The scan is the only expensive step in the F&O pipeline -- a yfinance batch
plus four NSE calls over a shortlist -- and its feed selection decides whether
the numbers are live or exchange-delayed. That choice is a data-integrity
gate, not a display preference, so it lives here and nowhere else: the web
cache and the trading agent both call `run_scan` instead of each carrying a
copy of the policy that can drift.

`run_scan` returns the `ScanResult` itself, not JSON -- the agent needs the
objects. Callers that want the printable payload call `report.as_dict` on it,
so the page and the CLI still render exactly the same numbers.
"""
from __future__ import annotations

from datetime import datetime

from trading.fno import config as C
from trading.fno.data import LiveFeed, ReplayFeed
from trading.fno.models import IST, ScanResult
from trading.fno.scanner import Scanner

DEFAULT_FYERS_BASE = "http://localhost:3001"


def pick_feed(*, replay: str | None = None, fyers: bool = False,
              allow_delayed: bool = True, fyers_base: str = DEFAULT_FYERS_BASE):
    """Choose the feed and say what the scan may do with it.

    Returns ``(feed, now, allow_delayed)``. Kept separate from `run_scan` so
    the policy is testable without opening a socket:

    * a replay is authoritative and self-describing -- it carries its own
      as-of time, and delayed data is the whole point of it;
    * a live broker feed is used when one is configured (``fyers``), when a
      token exists, or when the local Fyers bridge reports it is connected;
    * otherwise the free feed, where ``allow_delayed`` decides whether
      exchange-delayed candles may produce a signal at all.
    """
    if replay:
        feed = ReplayFeed(replay)
        return feed, feed.as_of, True

    from trading.fno.fyers import FyersClient, FyersFeed, load_token
    token, _ = load_token()
    client = FyersClient(base=fyers_base)
    is_connected, _ = client.status()
    if fyers or token is not None or is_connected:
        return FyersFeed(base=fyers_base), datetime.now(IST), True
    return LiveFeed(), datetime.now(IST), allow_delayed


def run_scan(*, symbols: list[str] | None = None, shortlist: int = C.SHORTLIST_SIZE,
             with_options: bool = True, replay: str | None = None,
             allow_delayed: bool = True, fyers: bool = False,
             fyers_base: str = DEFAULT_FYERS_BASE) -> ScanResult:
    """Run one scan and return it. No caching, no rendering, no side effects.

    ``with_options=False`` skips the NIFTY index-option chain read -- the
    agent wants stock candidates, not the chain, and the chain is one of the
    slower fetches.
    """
    feed, now, allow = pick_feed(replay=replay, fyers=fyers,
                                 allow_delayed=allow_delayed, fyers_base=fyers_base)
    return Scanner(feed, now, allow_delayed=allow, symbols=symbols,
                   shortlist=shortlist, with_options=with_options).run()
